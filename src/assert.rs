use std::fmt::Display;

use proc_macro2::TokenStream;
use syn::{parse::Parse, parse2};

pub struct Assert;

impl Assert {
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
}
