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

const MAX_PREFIX_LEN: usize = 20;
const LINE_LEN: usize = 100;

pub fn __equivalent(a: &str, b: &str) -> std::result::Result<(), String> {
    let a = simplify(a);
    let b = simplify(b);

    let min_length = a.len().min(b.len());
    let mut divergence_index: Option<usize> = None;
    for i in 0..min_length {
        if a[i] != b[i] {
            divergence_index = Some(i);
            break;
        }
    }

    let divergence_index: usize = match divergence_index {
        None => if a.len() == b.len() { return Ok(()) } else { min_length },
        Some(div) => div,
    };

    let start_index;
    let arrow_index;

    if MAX_PREFIX_LEN < divergence_index {
        start_index = divergence_index - MAX_PREFIX_LEN;
        arrow_index = MAX_PREFIX_LEN;
    } else {
        start_index = 0;
        arrow_index = divergence_index;
    }

    let mut div = String::with_capacity(arrow_index);
    for _ in 0..arrow_index { div.push(' ') }

    let mut left = String::new();
    for i in start_index..a.len().min(start_index + LINE_LEN) { left.push(a[i]); }

    let mut right = String::new();
    for i in start_index..b.len().min(start_index + LINE_LEN) { right.push(b[i]); }

    let message = format!(
        "assertion `left == right` failed\n  left: {}↓\n        {}\n right: {}↓\n        {}\n",
        div, left, div, right
    );

    Err(message)
}

fn simplify(text: &str) -> Vec<char> {
    let mut result = Vec::<char>::new();
    let mut previous = '\0';
    for c in text.chars() {
        if c.is_whitespace() {
            if previous != ' ' {
                result.push(' ');
            }
            previous = ' ';
        } else {
            if is_opening(previous) {
                result.push(' ');
            }
            else if previous != ' ' && is_closing(c) {
                result.push(' ');
            }
            result.push(c);
            previous = c;
        }
    }
    result
}

fn is_opening(c: char) -> bool {
    c == '(' || c == '[' || c == '{' 
}

fn is_closing(c: char) -> bool {
    c == ')' || c == ']' || c == '}' 
}