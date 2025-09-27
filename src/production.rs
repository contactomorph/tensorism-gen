use std::ops::Deref;

use proc_macro2::{Delimiter, Group, Ident, Literal, TokenStream, TokenTree};

use crate::analysis::types::{
    Discriminant, IndexingPosition, IndexingPositionContent, IndexingPositionEquivalence,
    IndexingPositionMapping,
};
use crate::model::header::RicciIndexer;
use crate::model::lambda::RicciLambda;
use crate::model::lambda::{RicciGroup, RicciSegment};
use crate::quote::ToTokens;

fn create_global_dim_identifier(index: &Ident) -> Ident {
    format_ident!("global_dim_for_{}", index)
}

fn create_local_dim_identifier(index: &Ident, discriminant: &Discriminant) -> Ident {
    format_ident!("local_dim_for_{}_{}", discriminant.deref(), index)
}

fn process_lambda(lambda: RicciLambda, discriminant: Discriminant, output: &mut TokenStream) {
    let mut body = TokenStream::new();
    process_segments(lambda.body.segments, discriminant.clone(), &mut body);
    let indexes = lambda.index_declaration.indexes.as_slice();

    if indexes.len() == 1 {
        let index = &indexes[0];
        let dimension_name = create_local_dim_identifier(index, &discriminant);
        let lambda_stream = quote! {(0usize..#dimension_name).map(|#index| { #body }) };
        output.extend(lambda_stream);
    } else {
        let indexes_tuple = quote! {(#(#indexes),*, )};
        let mut header = indexes_tuple.clone();

        for (i, index) in indexes.iter().enumerate() {
            let dimension_name = create_local_dim_identifier(index, &discriminant);
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

fn process_segments(
    segments: Vec<RicciSegment>,
    discriminant: Discriminant,
    output: &mut TokenStream,
) {
    let mut i = 0;
    for segment in segments {
        match segment {
            RicciSegment::SubGroup { delimiter, group } => {
                let mut content = TokenStream::new();
                let new_discriminant = discriminant.extend(i);
                i += 1;
                process_segments(group.segments, new_discriminant, &mut content);
                TokenTree::Group(Group::new(delimiter, content)).to_tokens(output);
            }
            RicciSegment::SubLambda(lambda) => {
                let new_discriminant = discriminant.extend(i);
                i += 1;
                process_lambda(*lambda, new_discriminant, output);
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

fn process_main_lambda(lambda: RicciLambda, discriminant: Discriminant, output: &mut TokenStream) {
    let dimensions = &lambda
        .index_declaration
        .indexes
        .iter()
        .map(create_global_dim_identifier)
        .collect::<Vec<_>>();
    let indexes = lambda.index_declaration.indexes;
    let mut substream = TokenStream::new();
    let order = dimensions.len();
    process_segments(lambda.body.segments, discriminant, &mut substream);
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

fn process_main_group(group: RicciGroup, discriminant: Discriminant, output: &mut TokenStream) {
    if group.segments.len() == 1 && matches!(group.segments[0], RicciSegment::SubLambda(_)) {
        for segment in group.segments {
            if let RicciSegment::SubLambda(lambda) = segment {
                process_main_lambda(*lambda, discriminant, output);
                return;
            }
        }
    } else {
        process_segments(group.segments, discriminant, output)
    }
}

fn produce_dimension_value(position: &IndexingPosition) -> TokenStream {
    match position.content {
        IndexingPositionContent::TensorSpecificIndex(pos) => {
            let tensor_name = &position.name;
            let pos = Literal::usize_unsuffixed(pos);
            quote! {
                ::ndarray::ArrayBase::<_, _>::dim(&#tensor_name).#pos
            }
        }
        IndexingPositionContent::TensorSingleIndex => {
            let tensor_name = &position.name;
            quote! { ::ndarray::ArrayBase::<_, _>::dim(&#tensor_name) }
        }
        _ => {
            todo!()
        }
    }
}

pub fn produce_prelude(mapping: IndexingPositionMapping) -> TokenStream {
    let mut content = TokenStream::new();
    let IndexingPositionMapping {
        equivalences,
        plain_values: _,
    } = mapping;
    for equivalence in equivalences {
        let IndexingPositionEquivalence { index, positions } = equivalence;
        match index {
            Some((index_name, discriminant)) => {
                for position in positions.iter().take(1) {
                    let value = produce_dimension_value(position);
                    let dimension_var = if discriminant.is_empty() {
                        create_global_dim_identifier(&index_name)
                    } else {
                        create_local_dim_identifier(&index_name, &discriminant)
                    };
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
    let mut content = produce_prelude(mapping);
    let discriminant = Discriminant::new();
    process_main_group(group, discriminant, &mut content);
    let mut output = TokenStream::new();
    TokenTree::Group(Group::new(Delimiter::Brace, content)).to_tokens(&mut output);
    output
}
