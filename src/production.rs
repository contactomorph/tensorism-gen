use proc_macro2::{Delimiter, Group, Literal, TokenStream, TokenTree};

use crate::analysis::types::{
    HeadKind, IndexingPosition, IndexingPositionEquivalence, IndexingPositionMapping,
};
use crate::model::header::RicciIndexer;
use crate::model::lambda::RicciLambda;
use crate::model::lambda::{RicciGroup, RicciSegment};
use crate::quote::ToTokens;

fn process_lambda(lambda: RicciLambda, output: &mut TokenStream) {
    let mut body = TokenStream::new();
    process_group(lambda.body, &mut body);
    let indexes = lambda.index_declaration.indexes.as_slice();

    if indexes.len() == 1 {
        let index = &indexes[0];
        let dimension_name = format_ident!("{}_dimension", index);
        let lambda_stream = quote! {(0usize..#dimension_name).map(|#index| { #body }) };
        output.extend(lambda_stream);
    } else {
        let indexes_tuple = quote! {(#(#indexes),*, )};
        let mut header = indexes_tuple.clone();

        for (i, index) in indexes.iter().enumerate() {
            let dimension_name = format_ident!("{}_dimension", index);
            header = if i == 0 {
                quote! {(0usize..#dimension_name).map(move |#index| #header)}
            } else {
                quote! {(0usize..#dimension_name).flat_map(move |#index| #header)}
            }
        }
        let lambda_stream = quote! { #header.map(|#indexes_tuple| { #body }) };
        output.extend(lambda_stream);
    }
}

fn process_segments(segments: Vec<RicciSegment>, output: &mut TokenStream) {
    for segment in segments.into_iter() {
        match segment {
            RicciSegment::SubGroup { delimiter, group } => {
                let mut content = TokenStream::new();
                process_group(*group, &mut content);
                TokenTree::Group(Group::new(delimiter, content)).to_tokens(output);
            }
            RicciSegment::SubLambda(lambda) => {
                process_lambda(*lambda, output);
            }
            RicciSegment::TensorCall {
                tensor_name,
                indexers,
            } => {
                let stream = if indexers.len() == 1 {
                    let indexer = indexers.into_iter().next().unwrap();
                    match indexer {
                        RicciIndexer::Direct { index } => {
                            quote! {
                                (* unsafe{ ::ndarray::ArrayBase::< _, _ >::uget(& #tensor_name, #index) })
                            }
                        }
                        _ => {
                            todo!()
                        }
                    }
                } else {
                    let mut indexes = Vec::<syn::Ident>::new();
                    for indexer in indexers.into_iter() {
                        match indexer {
                            RicciIndexer::Direct { index } => indexes.push(index),
                            _ => todo!(),
                        };
                    }

                    quote! {
                        (* unsafe{ ::ndarray::ArrayBase::< _, _ >::uget(& #tensor_name, (#(#indexes, )*)) })
                    }
                };
                output.extend(stream);
            }
            RicciSegment::Token(token) => {
                token.to_tokens(output);
            }
        }
    }
}

fn process_main_lambda(lambda: RicciLambda, output: &mut TokenStream) {
    let dimensions = &lambda
        .index_declaration
        .indexes
        .iter()
        .map(|i| format_ident!("{}_dimension", i))
        .collect::<Vec<_>>();
    let indexes = lambda.index_declaration.indexes;
    let mut substream = TokenStream::new();
    let order = dimensions.len();
    process_segments(lambda.body.segments, &mut substream);
    if order == 1 {
        let dimension = &dimensions[0];
        let index = &indexes[0];
        quote! {
            ::ndarray::Array::<_, ::ndarray::Dim<[::ndarray::Ix; 1usize]>>::from_shape_fn(
                #dimension,
                |#index| { #substream }
            )
        }
        .to_tokens(output);
    } else {
        quote! {
            ::ndarray::Array::<_, ::ndarray::Dim<[::ndarray::Ix; #order]>>::from_shape_fn(
                (#(#dimensions),*, ),
                |(#(#indexes),*, )| { #substream }
            )
        }
        .to_tokens(output);
    }
}

fn process_group(group: RicciGroup, output: &mut TokenStream) {
    if group.segments.len() == 1 && matches!(group.segments[0], RicciSegment::SubLambda(_)) {
        for segment in group.segments {
            if let RicciSegment::SubLambda(lambda) = segment {
                process_main_lambda(*lambda, output)
            }
        }
    } else {
        process_segments(group.segments, output)
    }
}

fn produce_dimension_value(position: IndexingPosition) -> TokenStream {
    match position.kind {
        HeadKind::Indexer => {
            todo!()
        }
        HeadKind::Tensor => {
            let tensor_name = position.name;
            if position.rank == 1 {
                quote! { ::ndarray::ArrayBase::<_, _>::dim(&#tensor_name) }
            } else {
                let pos = Literal::usize_unsuffixed(position.position);
                quote! {
                    ::ndarray::ArrayBase::<_, _>::dim(&#tensor_name).#pos
                }
            }
        }
    }
}

pub fn produce_header(mapping: IndexingPositionMapping) -> TokenStream {
    let mut content = TokenStream::new();
    let IndexingPositionMapping {
        equivalences,
        plain_values: _,
    } = mapping;
    for equivalence in equivalences {
        let IndexingPositionEquivalence { index, positions } = equivalence;
        match index {
            Some((index_name, _dimension_name)) => {
                for position in positions {
                    let value = produce_dimension_value(position);
                    let dimension_var = format_ident!("{}_dimension", index_name);
                    let definition = quote! {
                        let #dimension_var = #value;
                    };
                    content.extend(definition);
                }
            }
            None => {
                todo!()
            }
        }
    }
    content
}

pub fn produce(group: RicciGroup, mapping: IndexingPositionMapping) -> TokenStream {
    let mut content = produce_header(mapping);
    process_group(group, &mut content);
    let mut output = TokenStream::new();
    TokenTree::Group(Group::new(Delimiter::Brace, content)).to_tokens(&mut output);
    output
}
