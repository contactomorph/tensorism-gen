#![feature(proc_macro_quote)]
#![feature(extend_one)]

extern crate proc_macro;
extern crate syn;
#[macro_use]
extern crate quote;

use proc_macro2::{Literal, TokenStream, TokenTree};

mod types;
mod parsing;
mod sequentialization;

use parsing::parse;
use sequentialization::sequentialize;

#[proc_macro]
pub fn tensorism_make(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    match parse(input) {
        Err(invalid_stream) => invalid_stream.into(),
        Ok((func, index_use)) => sequentialize(func, index_use).into(),
    }
}

#[proc_macro]
pub fn tensorism_string_for_make(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    match parse(input) {
        Err(invalid_stream) => invalid_stream.into(),
        Ok((func, index_use)) => {
            let output = sequentialize(func, index_use);
            let string = output.to_string();
            let mut output = TokenStream::new();
            output.extend_one(TokenTree::Literal(Literal::string(string.as_str())));
            output.into()
        }
    }
}
