use std::collections::HashMap;

use crate::analysis::types::{
    AliasSource, IncreasingInteger, IndexingPosition, IndexingPositionContent,
    IndexingPositionEquivalence, IndexingPositionMapping, PositionalPlainValue,
};
use crate::model::header::{RicciAliasDeclaration, RicciIndexer};
use crate::model::lambda::{RicciLambda, RicciSegment};
use crate::quote::ToTokens;
use crate::top_group::TopGroup;
use crate::unification::{add_unification_for_group, add_unification_for_lambda};
use proc_macro2::{Delimiter, Group, Ident, Literal, Punct, TokenStream, TokenTree};

pub struct ProductionCollector {
    free_ricci_number: IncreasingInteger,
    free_plain_number: IncreasingInteger,
    ricci_numbers_per_index: HashMap<Ident, usize>,
}

impl ProductionCollector {
    pub fn new() -> Self {
        Self {
            free_ricci_number: IncreasingInteger::new(),
            free_plain_number: IncreasingInteger::new(),
            ricci_numbers_per_index: HashMap::new(),
        }
    }

    pub fn upsert_ricci_number(&mut self, index: &Ident) -> usize {
        let ricci_number = self.free_ricci_number.get_next();
        self.ricci_numbers_per_index
            .insert(index.clone(), ricci_number);
        ricci_number
    }

    pub fn get_ricci_number(&self, index: &Ident) -> Option<usize> {
        self.ricci_numbers_per_index.get(index).copied()
    }
    pub fn upsert_plain_number(&mut self) -> usize {
        self.free_plain_number.get_next()
    }
}

fn create_dim_identifier(ricci_number: usize) -> Ident {
    format_ident!("tsm_dim_{}", ricci_number)
}

fn create_plain_identifier(plain_number: usize) -> Ident {
    format_ident!("tsm_plain_{}", plain_number)
}

fn create_ptr_identifier(tensor_name: &Ident) -> Ident {
    format_ident!("tsm_ptr_{}", tensor_name)
}

fn create_stride_identifier(i: usize, tensor_name: &Ident) -> Ident {
    format_ident!("tsm_stride_{}_{}", i, tensor_name)
}

fn create_reindexing_type(rank: usize) -> Ident {
    format_ident!("Reindexing{}", rank)
}

fn process_indexer(
    indexer: RicciIndexer,
    collector: &mut ProductionCollector,
    already_wrapped_by_unsafe: bool,
) -> TokenStream {
    match indexer {
        RicciIndexer::Direct {
            index: source_index,
        } => {
            quote! { #source_index }
        }
        RicciIndexer::Reindexing {
            reindexing_name,
            indexers,
        } => {
            let mut indexers_streams = Vec::<TokenStream>::new();
            for indexer in indexers {
                indexers_streams.push(process_indexer(indexer, collector, true))
            }
            let reindexer_type = create_reindexing_type(indexers_streams.len());
            if already_wrapped_by_unsafe {
                quote! {
                    ::tensorism::#reindexer_type::get_unchecked( &#reindexing_name, #(#indexers_streams),* )
                }
            } else {
                quote! {
                    unsafe { ::tensorism::#reindexer_type::get_unchecked( &#reindexing_name, #(#indexers_streams),* ) }
                }
            }
        }
        RicciIndexer::Reverse {
            index: source_index,
        } => {
            let ricci_number = collector.get_ricci_number(&source_index).unwrap();
            let dim = create_dim_identifier(ricci_number);
            quote! { (#dim - 1 - #source_index) }
        }
        RicciIndexer::Plain { .. } => {
            let plain_number = collector.upsert_plain_number();
            let plain = create_plain_identifier(plain_number);
            quote! { #plain }
        }
    }
}

fn process_indexers(
    indexers: Vec<RicciIndexer>,
    tensor_name: &Ident,
    collector: &mut ProductionCollector,
) -> TokenStream {
    let mut indexer_stream = TokenStream::new();
    for (i, indexer) in indexers.into_iter().enumerate() {
        let indexer_content = process_indexer(indexer, collector, true);
        let stride_name = create_stride_identifier(i, tensor_name);
        if 0 < i {
            TokenTree::Punct(Punct::new('+', proc_macro2::Spacing::Alone))
                .to_tokens(&mut indexer_stream);
        }
        indexer_stream.extend(quote! { (#indexer_content as isize) * #stride_name });
    }
    indexer_stream
}

fn process_alias_declarations(
    alias_declarations: Vec<RicciAliasDeclaration>,
    collector: &mut ProductionCollector,
    output: &mut TokenStream,
) {
    for declaration in alias_declarations {
        let index = declaration.index;
        let indexer = process_indexer(declaration.indexer, collector, false);
        output.extend(quote! { let #index = #indexer; });
    }
}

fn process_lambda(
    lambda: RicciLambda,
    collector: &mut ProductionCollector,
    output: &mut TokenStream,
) {
    let mut body = TokenStream::new();
    process_alias_declarations(lambda.alias_declarations, collector, &mut body);
    process_segments(lambda.body.segments, collector, &mut body);
    let indexes = lambda.index_declaration.indexes.as_slice();

    if indexes.len() == 1 {
        let index = &indexes[0];
        let ricci_number = collector.upsert_ricci_number(index);
        let dimension_name = create_dim_identifier(ricci_number);

        let lambda_stream = match lambda.filter {
            Some(filter) => {
                let mut condition = TokenStream::new();
                process_segments(filter.segments, collector, &mut condition);
                quote! {(0usize..#dimension_name).filter(|&#index| { #condition }).map(|#index| { #body }) }
            }
            None => {
                quote! {(0usize..#dimension_name).map(|#index| { #body }) }
            }
        };
        output.extend(lambda_stream);
    } else {
        let indexes_tuple = quote! {(#(#indexes,)* )};
        let mut header = indexes_tuple.clone();

        for (i, index) in indexes.iter().enumerate() {
            let ricci_number = collector.upsert_ricci_number(index);
            let dimension_name = create_dim_identifier(ricci_number);

            header = if i == 0 {
                quote! {(0usize..#dimension_name).map(move |#index| { #header })}
            } else {
                quote! {(0usize..#dimension_name).flat_map(move |#index| { #header })}
            }
        }
        let lambda_stream = match lambda.filter {
            Some(filter) => {
                let mut condition = TokenStream::new();
                process_segments(filter.segments, collector, &mut condition);
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
    collector: &mut ProductionCollector,
    output: &mut TokenStream,
) {
    for segment in segments {
        match segment {
            RicciSegment::SubGroup { delimiter, group } => {
                let mut content = TokenStream::new();
                process_segments(group.segments, collector, &mut content);
                TokenTree::Group(Group::new(delimiter, content)).to_tokens(output);
            }
            RicciSegment::SubLambda(lambda) => {
                process_lambda(*lambda, collector, output);
            }
            RicciSegment::TensorCall {
                tensor_name,
                indexers,
            } => {
                let indexer_stream = process_indexers(indexers, &tensor_name, collector);
                let ptr_name = create_ptr_identifier(&tensor_name);
                let stream = quote! {
                    (*unsafe { &*#ptr_name.offset(#indexer_stream) })
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
    collector: &mut ProductionCollector,
    output: &mut TokenStream,
) {
    if lambda.filter.is_some() {
        panic!("Macro level lambda cannot have a filter.");
    }

    let mut body = TokenStream::new();
    process_alias_declarations(lambda.alias_declarations, collector, &mut body);
    process_segments(lambda.body.segments, collector, &mut body);

    let mut indexes = lambda.index_declaration.indexes;
    let mut first = true;
    indexes.reverse();
    for index in indexes {
        let dimension = create_dim_identifier(collector.ricci_numbers_per_index[&index]);
        if first {
            body = quote! {
                for #index in 0usize..#dimension {
                    let tsm_value = { #body };
                    unsafe {
                        tsm_res_ptr.write(tsm_value);
                        tsm_res_ptr = tsm_res_ptr.add(1);
                    }
                };
            };
            first = false;
        } else {
            body = quote! {
                for #index in 0usize..#dimension {
                    #body
                };
            };
        }
    }
    output.extend(body);
}

fn produce_dimension_value(position: &IndexingPosition) -> TokenStream {
    let name = &position.name;
    match position.content {
        IndexingPositionContent::TensorIndex(pos, rank) => {
            let pos = Literal::usize_unsuffixed(pos);
            if rank == 1 {
                quote! {
                    ::ndarray::ArrayBase::<_, _>::dim(&#name)
                }
            } else {
                quote! {
                    ::ndarray::ArrayBase::<_, _>::dim(&#name).#pos
                }
            }
        }
        IndexingPositionContent::IndexerResult(rank) => {
            let reindexer_type = create_reindexing_type(rank);
            quote! {
                ::tensorism::#reindexer_type::get_output_bound(& #name)
            }
        }
        IndexingPositionContent::IndexerIndex(pos, rank) => {
            let reindexer_type = create_reindexing_type(rank);
            let get_input_function = format_ident!("get_input{}_bound", pos);
            quote! {
                ::tensorism::#reindexer_type::#get_input_function(& #name)
            }
        }
    }
}

fn produce_dimension_value_for_alias(alias_source: &AliasSource) -> TokenStream {
    match alias_source {
        AliasSource::FromReindexing {
            reindexing_name,
            rank,
        } => {
            let reindexer_type = create_reindexing_type(*rank);
            quote! {
                ::tensorism::#reindexer_type::get_output_bound(& #reindexing_name)
            }
        }
        AliasSource::FromIndex { ricci_number } => {
            let dimension = create_dim_identifier(*ricci_number);
            quote! {
                #dimension
            }
        }
    }
}

fn format_indexer_result(reindexing_name: &Ident, index: &Ident, rank: usize) -> String {
    if 1 <= rank {
        format!(
            "{} = {}[{}_]",
            index,
            reindexing_name,
            "_, ".repeat(rank - 1)
        )
    } else {
        format!("{} = {}[]", index, reindexing_name)
    }
}

fn format_position(position: &IndexingPosition, index: &Ident) -> String {
    match position.content {
        IndexingPositionContent::IndexerResult(rank) => {
            format_indexer_result(&position.name, index, rank)
        }
        IndexingPositionContent::IndexerIndex(pos, rank) => {
            let beginning = "_, ".repeat(pos);
            let end = if pos + 1 < rank {
                ", _".repeat(rank - pos - 1)
            } else {
                String::new()
            };
            format!("{}[{}{}{}]", position.name, beginning, index, end)
        }
        IndexingPositionContent::TensorIndex(pos, rank) => {
            let beginning = "_, ".repeat(pos);
            let end = if pos + 1 < rank {
                ", _".repeat(rank - pos - 1)
            } else {
                String::new()
            };
            format!("{}[{}{}{}]", position.name, beginning, index, end)
        }
    }
}

fn format_alias_source(alias_source: &AliasSource, alias: &Ident) -> String {
    match alias_source {
        AliasSource::FromReindexing {
            reindexing_name,
            rank,
        } => format_indexer_result(reindexing_name, alias, *rank),
        AliasSource::FromIndex { .. } => {
            format!("{} = _", alias)
        }
    }
}

pub fn produce_prelude(
    equivalences: Vec<IndexingPositionEquivalence>,
    plain_values: Vec<PositionalPlainValue>,
    output: &mut TokenStream,
) {
    let mut tensors_with_ranks = HashMap::<Ident, usize>::new();
    let mut tensors = Vec::<Ident>::new();

    for equivalence in equivalences {
        let IndexingPositionEquivalence {
            ricci_number,
            alias_source,
            positions,
            index,
        } = equivalence;
        let mut maybe_dimension_var: Option<(Ident, String)> = None;
        if let Some(alias_source) = alias_source {
            let value = produce_dimension_value_for_alias(&alias_source);
            let position = format_alias_source(&alias_source, &index);
            let dimension_var = create_dim_identifier(ricci_number);
            let definition = quote! {
                let #dimension_var = #value;
            };
            output.extend(definition);
            maybe_dimension_var = Some((dimension_var, position));
        }
        for position in positions.iter() {
            if let IndexingPositionContent::TensorIndex(_, rank) = position.content
                && tensors_with_ranks
                    .insert(position.name.clone(), rank)
                    .is_none()
            {
                tensors.push(position.name.clone());
            }

            let value = produce_dimension_value(position);
            let position = format_position(position, &index);
            match &maybe_dimension_var {
                None => {
                    let dimension_var = create_dim_identifier(ricci_number);
                    let definition = quote! {
                        let #dimension_var = #value;
                    };
                    output.extend(definition);
                    maybe_dimension_var = Some((dimension_var, position));
                }
                Some((dimension_var, previous_position)) => {
                    let message = format!(
                        "Dimensions are not matching between {} and {}",
                        previous_position, position
                    );
                    let message = Literal::string(message.as_str());
                    let consistency_check = quote! {
                        if #dimension_var != #value {
                            panic!(#message);
                        };
                    };
                    output.extend(consistency_check);
                }
            }
        }
    }

    for plain_value in plain_values {
        if let IndexingPositionContent::TensorIndex(_, rank) = plain_value.position.content
            && tensors_with_ranks
                .insert(plain_value.position.name.clone(), rank)
                .is_none()
        {
            tensors.push(plain_value.position.name.clone());
        }

        let plain = create_plain_identifier(plain_value.plain_number);
        let expr = plain_value.expr;
        let value = produce_dimension_value(&plain_value.position);
        let index = Ident::new("plain", proc_macro2::Span::call_site());
        let position = format_position(&plain_value.position, &index);
        let message = format!("Plain value is out of bounds in {}", position);
        let plain_value_declaration = quote! {
            let #plain: usize = #expr;
            if #plain >= #value {
                panic!(#message);
            };
        };
        output.extend(plain_value_declaration);
    }

    tensors.sort();

    for tensor_name in tensors.into_iter() {
        let rank = tensors_with_ranks[&tensor_name];
        let ptr_name = create_ptr_identifier(&tensor_name);
        let ptr_declaration = quote! {
            let #ptr_name: *const _ = #tensor_name.as_ptr();
            let tsm_strides = #tensor_name.strides();
        };
        output.extend(ptr_declaration);
        for i in 0..rank {
            let stride_name = create_stride_identifier(i, &tensor_name);
            let stride_declaration = quote! {
                let #stride_name: isize = tsm_strides[#i];
            };
            output.extend(stride_declaration);
        }
    }
}

pub fn produce(top_group: TopGroup, mapping: IndexingPositionMapping) -> TokenStream {
    let IndexingPositionMapping {
        equivalences,
        plain_values,
    } = mapping;

    let mut content = TokenStream::new();

    produce_prelude(equivalences, plain_values, &mut content);

    match top_group {
        TopGroup::Group(group) => {
            let mut collector = ProductionCollector::new();
            add_unification_for_group(&group, &mut content);
            process_segments(group.segments, &mut collector, &mut content);
        }
        TopGroup::Lambda(lambda) => {
            let mut collector = ProductionCollector::new();
            let mut dimensions = Vec::<Ident>::new();
            for index in &lambda.index_declaration.indexes {
                collector.upsert_ricci_number(index);
                let dimension = create_dim_identifier(collector.ricci_numbers_per_index[index]);
                dimensions.push(dimension);
            }
            let order = Literal::usize_suffixed(lambda.index_declaration.indexes.len());
            let ptr_declaration = quote! {
                type TsmDimensionType = ::ndarray::Dim<[::ndarray::Ix;#order]>;
                let mut tsm_res = ::ndarray::Array::<_,TsmDimensionType>::uninit((#(#dimensions,)*));
                let mut tsm_res_ptr = tsm_res.as_mut_ptr() as *mut _;
            };
            content.extend(ptr_declaration);
            add_unification_for_lambda(&lambda, &mut content);
            process_main_lambda(*lambda, &mut collector, &mut content);
            content.extend(quote! { unsafe { tsm_res.assume_init() } });
        }
    }
    let mut output = TokenStream::new();
    TokenTree::Group(Group::new(Delimiter::Brace, content)).to_tokens(&mut output);
    output
}
