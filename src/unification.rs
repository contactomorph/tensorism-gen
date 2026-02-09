use std::collections::HashSet;

use proc_macro2::{Group, Literal, TokenStream, TokenTree};
use quote::ToTokens;
use syn::Ident;

use crate::model::lambda::{RicciGroup, RicciLambda, RicciSegment};

pub struct UnificationCollector {
    indexes: HashSet<Ident>,
}

impl UnificationCollector {
    pub fn new() -> Self {
        Self {
            indexes: HashSet::new(),
        }
    }

    pub fn add_index(&mut self, index: &Ident) {
        self.indexes.insert(index.clone());
    }

    pub fn forget_index(&mut self, index: &Ident) {
        self.indexes.remove(index);
    }
}

fn create_ptr_identifier(tensor_name: &Ident) -> Ident {
    format_ident!("tsm_ptr_{}", tensor_name)
}

fn create_unification_lambda_body(
    lambda: &RicciLambda,
    collector: &mut UnificationCollector,
) -> TokenStream {
    let mut body = TokenStream::new();
    let mut new_indexes = Vec::new();
    for index in &lambda.index_declaration.indexes {
        collector.add_index(index);
        new_indexes.push(index.clone());
    }
    for declaration in &lambda.alias_declarations {
        collector.add_index(&declaration.index);
        new_indexes.push(declaration.index.clone());
    }
    create_unification_segments(&lambda.body.segments, collector, &mut body);
    for index in new_indexes {
        collector.forget_index(&index);
    }
    body
}

fn create_unification_segments(
    segments: &[RicciSegment],
    collector: &mut UnificationCollector,
    output: &mut TokenStream,
) {
    for segment in segments {
        match segment {
            RicciSegment::SubGroup { delimiter, group } => {
                let mut content = TokenStream::new();
                create_unification_segments(&group.segments, collector, &mut content);
                TokenTree::Group(Group::new(*delimiter, content)).to_tokens(output);
            }
            RicciSegment::SubLambda(lambda) => {
                let body = create_unification_lambda_body(lambda, collector);
                quote! {(0usize..).map(|_| { #body })}.to_tokens(output);
            }
            RicciSegment::TensorCall { tensor_name, .. } => {
                let ptr_name = create_ptr_identifier(tensor_name);
                let stream = quote! {
                    (*unsafe{ &*#ptr_name })
                };
                output.extend(stream);
            }
            RicciSegment::Token(token) => {
                if let TokenTree::Ident(ident) = token {
                    if collector.indexes.contains(ident) {
                        TokenTree::Literal(Literal::usize_suffixed(0)).to_tokens(output);
                    } else {
                        token.to_tokens(output);
                    }
                } else {
                    token.to_tokens(output);
                }
            }
        }
    }
}

pub fn add_unification_for_group(group: &RicciGroup, output: &mut TokenStream) {
    let mut collector = UnificationCollector::new();
    let mut body = TokenStream::new();
    create_unification_segments(&group.segments, &mut collector, &mut body);
    output.extend(quote! {
        fn tsm_unify<T, D>(
            _tensor: &::ndarray::Array::<std::mem::MaybeUninit<T>, D>,
            _ptr: *mut T,
            _f: impl Fn() -> T,
        ) {}
        tsm_unify(&tsm_res, tsm_res_ptr, || { #body });
    });
}

pub fn add_unification_for_lambda(lambda: &RicciLambda, output: &mut TokenStream) {
    let mut collector = UnificationCollector::new();
    let body = create_unification_lambda_body(lambda, &mut collector);
    output.extend(quote! {
        fn tsm_unify<T, D>(
            _tensor: &::ndarray::Array::<std::mem::MaybeUninit<T>, D>,
            _ptr: *mut T,
            _f: impl Fn() -> T,
        ) {}
        tsm_unify(&tsm_res, tsm_res_ptr, || { #body });
    });
}
