#![feature(proc_macro_quote)]
#![feature(extend_one)]

extern crate proc_macro;
extern crate syn;
#[macro_use]
extern crate quote;

use proc_macro2::{Literal, TokenStream, TokenTree};

mod parsing;
mod sequentialization;
mod types;

use parsing::parse;
use sequentialization::sequentialize;

fn simplify(text: &String) -> String {
    let mut result = String::new();
    text.split('\n').map(|s| s.trim()).for_each(|s| {
        result.extend_one(s);
        result.extend_one(' ')
    });
    result
}

#[proc_macro]
pub fn make(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    match parse(input) {
        Err(invalid_stream) => invalid_stream.into(),
        Ok((func, index_use)) => sequentialize(func, index_use).into(),
    }
}

#[proc_macro]
pub fn format_for_make(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    match parse(input) {
        Err(invalid_stream) => invalid_stream.into(),
        Ok((func, index_use)) => {
            let output = sequentialize(func, index_use);
            let string = simplify(&output.to_string());
            let mut output = TokenStream::new();
            output.extend_one(TokenTree::Literal(Literal::string(string.as_str())));
            output.into()
        }
    }
}
