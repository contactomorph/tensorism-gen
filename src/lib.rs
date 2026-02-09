//! Macro for handling arrays with multiple indexes.
//!
//! This crate should not be used directly. Instead, use the `tensorism` crate.
//! It re-exports the items defined here and provides additional functionality.
extern crate proc_macro;
#[macro_use]
extern crate quote;

use proc_macro2::{Literal, TokenStream, TokenTree};

mod analysis;
mod model;
mod production;
mod unification;

use quote::ToTokens;

use crate::analysis::{inspection::inspect, top_group};

fn simplify(text: &str) -> String {
    let mut result = String::new();
    text.split('\n').map(|s| s.trim()).for_each(|s| {
        result.push_str(s);
        result.push(' ')
    });
    result
}

/// Macro that generate a new `ndarray::Array` by evaluating a special domain-specific language used for its argument.
/// See the `tensorism` crate for documentation and examples.
#[proc_macro]
pub fn new_ndarray(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    match syn::parse2::<crate::model::lambda::RicciGroup>(input.into()) {
        Err(error) => {
            let message = format!("Failed to parse input: {}", error);
            quote! { compile_error!(#message) }.into()
        }
        Ok(group) => match inspect(group) {
            Err(error) => {
                let message = format!("Index issues: {}", error);
                quote! { compile_error!(#message) }.into()
            }
            Ok((top_group, mapping)) => production::produce(top_group, mapping).into(),
        },
    }
}

#[doc(hidden)]
#[proc_macro]
pub fn format_new_ndarray(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    match syn::parse2::<crate::model::lambda::RicciGroup>(input.into()) {
        Err(error) => {
            let message = format!("Failed to parse input: {}", error);
            quote! { compile_error!(#message) }.into()
        }
        Ok(group) => match inspect(group) {
            Err(error) => {
                let message = format!("Index issues: {}", error);
                quote! { compile_error!(#message) }.into()
            }
            Ok((top_group, mapping)) => {
                let output = production::produce(top_group, mapping);
                let string = simplify(&output.to_string());
                let mut output = TokenStream::new();
                TokenTree::Literal(Literal::string(string.as_str())).to_tokens(&mut output);
                output.into()
            }
        },
    }
}
