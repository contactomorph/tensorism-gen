use std::fmt::Display;
use proc_macro2::TokenStream;
use syn::{parse::Parse, parse2};

pub fn parse_and_display<T>(tokens: TokenStream) -> String
where
    T: Parse + Display,
{
    match parse2::<T>(tokens) {
        Ok(item) => format!("{}", item),
        Err(error) => {
            let message = format!(
                "Failed to parse type `{}`: {}",
                std::any::type_name::<T>(),
                error
            );
            message
        }
    }
}

#[macro_export]
macro_rules! equivalent {
    ($a: expr, $b: expr) => {
        match $crate::__equivalent($a, $b) {
            Ok(_) => {},
            Err(message) => panic!("{}", message),
        }
    }
}

pub fn __equivalent(a: &str, b: &str) -> std::result::Result<(), String> {
    let mut a = decompose(a);
    let mut b = decompose(b);

    loop {
        let x = a.next();
        let y = b.next();
        match (x, y) {
            (None, None) => break Ok(()),
            (None, Some(y)) => {
                let message = format!(
                    "assertion `left == right` failed\n  left:\n right: {}\n",
                    plug(y, b)
                );
                break Err(message)
            }
            (Some(x), None) => {
                let message = format!(
                    "assertion `left == right` failed\n  left: {}\n right:\n",
                    plug(x, a)
                );
                break Err(message)
            }
            (Some(x), Some(y)) => {
                if x != y {
                    let message = format!(
                        "assertion `left == right` failed\n  left: {}\n right: {}\n",
                        plug(x, a),
                        plug(y, b)
                    );
                    break Err(message)
                }
            }
        }
    }
}

const THRESHOLD: usize = 100;

fn plug<'a>(first: &'a str, rest: impl Iterator<Item=&'a str>) -> String {
    let mut result = String::new();
    result.push_str(first);
    for part in rest {
        if result.len() >= THRESHOLD {
            break;
        }
        result.push(' ');
        result.push_str(part);
    }
    result
}

fn decompose(text: &str) -> impl Iterator<Item=&str> {
    text.split_whitespace().flat_map(decompose_block)
}

fn decompose_block(text: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut last = 0;
    for (index, matched) in text.match_indices(|c: char| c.is_ascii_punctuation()) {
        if last != index {
            result.push(&text[last..index]);
        }
        result.push(matched);
        last = index + matched.len();
    }
    if last < text.len() {
        result.push(&text[last..]);
    }
    result
}