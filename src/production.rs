use proc_macro2::{Delimiter, Group, Ident, Literal, TokenStream, TokenTree};

use crate::analysis::types::{
    IncreasingInteger, IndexingPosition, IndexingPositionContent, IndexingPositionEquivalence,
    IndexingPositionMapping,
};
use crate::model::header::RicciIndexer;
use crate::model::lambda::RicciLambda;
use crate::model::lambda::{RicciGroup, RicciSegment};
use crate::quote::ToTokens;

fn create_dim_identifier(ricci_number: usize) -> Ident {
    format_ident!("dim_number_{}", ricci_number)
}

fn process_lambda(
    lambda: RicciLambda,
    free_ricci_number: &mut IncreasingInteger,
    output: &mut TokenStream,
) {
    let mut body = TokenStream::new();
    process_segments(lambda.body.segments, free_ricci_number, &mut body);
    let indexes = lambda.index_declaration.indexes.as_slice();

    if indexes.len() == 1 {
        let index = &indexes[0];
        let dimension_name = create_dim_identifier(free_ricci_number.get_next());

        let lambda_stream = match lambda.filter {
            Some(filter) => {
                let mut condition = TokenStream::new();
                process_segments(filter.segments, free_ricci_number, &mut condition);
                quote! {(0usize..#dimension_name).filter(|&#index| { #condition }).map(|#index| { #body }) }
            }
            None => {
                quote! {(0usize..#dimension_name).map(|#index| { #body }) }
            }
        };
        output.extend(lambda_stream);
    } else {
        let indexes_tuple = quote! {(#(#indexes),*, )};
        let mut header = indexes_tuple.clone();

        for (i, index) in indexes.iter().enumerate() {
            let dimension_name = create_dim_identifier(free_ricci_number.get_next());

            header = if i == 0 {
                quote! {(0usize..#dimension_name).map(move |#index| { #header })}
            } else {
                quote! {(0usize..#dimension_name).flat_map(move |#index| { #header })}
            }
        }
        let lambda_stream = match lambda.filter {
            Some(filter) => {
                let mut condition = TokenStream::new();
                process_segments(filter.segments, free_ricci_number, &mut condition);
                quote! { #header.filter(|&#indexes_tuple| { #condition }).map(|#indexes_tuple| { #body }) }
            }
            None => {
                quote! { #header.map(|#indexes_tuple| { #body }) }
            }
        };
        output.extend(lambda_stream);
    }
}

fn process_segments(
    segments: Vec<RicciSegment>,
    free_ricci_number: &mut IncreasingInteger,
    output: &mut TokenStream,
) {
    for segment in segments {
        match segment {
            RicciSegment::SubGroup { delimiter, group } => {
                let mut content = TokenStream::new();
                process_segments(group.segments, free_ricci_number, &mut content);
                TokenTree::Group(Group::new(delimiter, content)).to_tokens(output);
            }
            RicciSegment::SubLambda(lambda) => {
                process_lambda(*lambda, free_ricci_number, output);
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

fn process_main_lambda(
    lambda: RicciLambda,
    mut free_ricci_number: IncreasingInteger,
    output: &mut TokenStream,
) {
    if lambda.filter.is_some() {
        panic!("Macro level lambda cannot have a filter.");
    }
    let dimensions = &lambda
        .index_declaration
        .indexes
        .iter()
        .map(|_| create_dim_identifier(free_ricci_number.get_next()))
        .collect::<Vec<_>>();
    let indexes = lambda.index_declaration.indexes;
    let mut substream = TokenStream::new();
    let order = dimensions.len();
    process_segments(lambda.body.segments, &mut free_ricci_number, &mut substream);
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

fn process_main_group(group: RicciGroup, output: &mut TokenStream) {
    let mut free_ricci_number = IncreasingInteger::new();
    if group.segments.len() == 1 && matches!(group.segments[0], RicciSegment::SubLambda(_)) {
        for segment in group.segments {
            if let RicciSegment::SubLambda(lambda) = segment {
                process_main_lambda(*lambda, free_ricci_number, output);
                return;
            }
        }
    } else {
        process_segments(group.segments, &mut free_ricci_number, output)
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

pub fn produce_prelude(equivalences: Vec<IndexingPositionEquivalence>) -> TokenStream {
    let mut content = TokenStream::new();
    for equivalence in equivalences {
        let IndexingPositionEquivalence {
            ricci_number,
            positions,
        } = equivalence;
        let mut maybe_dimension_var: Option<Ident> = None;
        for position in positions.iter() {
            let value = produce_dimension_value(position);
            match &maybe_dimension_var {
                None => {
                    let dimension_var = create_dim_identifier(ricci_number);
                    let definition = quote! {
                        let #dimension_var = #value;
                    };
                    content.extend(definition);
                    maybe_dimension_var = Some(dimension_var);
                }
                Some(dimension_var) => {
                    let consistency_check = quote! {
                        if #dimension_var != #value {
                            panic!("Dimensions are not matching");
                        }
                    };
                    content.extend(consistency_check);
                }
            }
        }
    }
    content
}

pub fn produce(group: RicciGroup, mapping: IndexingPositionMapping) -> TokenStream {
    let IndexingPositionMapping { equivalences, .. } = mapping;
    let mut content = produce_prelude(equivalences);
    process_main_group(group, &mut content);
    let mut output = TokenStream::new();
    TokenTree::Group(Group::new(Delimiter::Brace, content)).to_tokens(&mut output);
    output
}
