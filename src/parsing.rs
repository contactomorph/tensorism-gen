use proc_macro2::{Delimiter, Group, Ident, Punct, Span, TokenStream, TokenTree};
use crate::types::*;

fn parse_punct(
    punct: &Punct,
    tokens: &mut impl Iterator<Item = TokenTree>,
    func: &mut EinsteinFunction,
    index_use: &mut IndexUse,
) -> Result<(), TokenStream> {
    let c = punct.as_char();
    if c == ';' {
        let invalid_stream = quote_spanned! {
            punct.span() => compile_error!("Character ';' is forbidden")
        };
        return Err(invalid_stream);
    } else if c == '$' {
        let inverted_indexes = func.retrieve_all_indexes();
        let mut new_einst = EinsteinFunction::new(punct.span(), inverted_indexes);
        parse_tensor_func(tokens, &mut new_einst, index_use)?;
        func.content.push(EinsteinAlternative::Func(new_einst));
    } else {
        func.push_token(TokenTree::Punct(punct.clone()));
    }
    Ok(())
}

fn parse_tensor_indexing(group: &Group) -> Result<Vec<Ident>, TokenStream> {
    let mut indexes = Vec::new();
    for token in group.stream() {
        match token {
            TokenTree::Ident(index) => indexes.push(index),
            TokenTree::Punct(p) => {
                if p.as_char() != ',' {
                    let invalid_stream = quote_spanned! {
                        group.span() => compile_error!("Invalid content in indexes")
                    };
                    return Err(invalid_stream);
                }
            }
            _ => {
                let invalid_stream = quote_spanned! {
                    group.span() => compile_error!("Invalid content in indexes")
                };
                return Err(invalid_stream);
            }
        }
    }
    Ok(indexes)
}

fn parse_group(
    group: &Group,
    func: &mut EinsteinFunction,
    index_use: &mut IndexUse,
) -> Result<(), TokenStream> {
    match group.delimiter() {
        Delimiter::Brace => {
            let invalid_stream = quote_spanned! {
                group.span() => compile_error!("Characters '{' and '}' are forbidden")
            };
            return Err(invalid_stream);
        }
        Delimiter::Bracket => {
            if let Some(EinsteinAlternative::Tree(TokenTree::Ident(tensor_name))) =
                func.content.last()
            {
                let tensor_name = tensor_name.clone();
                let indexes = parse_tensor_indexing(group)?;
                for (position, index_name) in indexes.iter().enumerate() {
                    index_use.push(index_name.clone(), tensor_name.clone(), position)
                }
                func.content.pop();
                func.content.push(EinsteinAlternative::TensorAccess {
                    tensor_name,
                    span: group.span(),
                    indexes,
                });
            } else {
                let invalid_stream = quote_spanned! {
                    group.span() => compile_error!("Invalid tensor name: an identifier was expected")
                };
                return Err(invalid_stream);
            }
        }
        _ => {
            let mut inner_func = EinsteinFunction::new(group.span(), Vec::new());
            parse_tensor_func(&mut group.stream().into_iter(), &mut inner_func, index_use)?;
            let group = EinsteinAlternative::ParensGroup(group.span(), inner_func.content);
            func.content.push(group);
        }
    }
    Ok(())
}

fn parse_tensor_func(
    tokens: &mut impl Iterator<Item = TokenTree>,
    func: &mut EinsteinFunction,
    index_use: &mut IndexUse,
) -> Result<(), TokenStream> {
    while let Some(token) = tokens.next() {
        match token {
            TokenTree::Punct(ref punct) => parse_punct(punct, tokens, func, index_use)?,
            TokenTree::Group(ref group) => parse_group(group, func, index_use)?,
            TokenTree::Ident(ref ident) => func.push_token(TokenTree::Ident(ident.clone())),
            TokenTree::Literal(_) => func.push_token(token),
        }
    }
    Ok(())
}

pub fn parse(
    input: proc_macro::TokenStream,
) -> Result<(EinsteinFunction, IndexUse), TokenStream> {
    let input: TokenStream = input.into();
    let mut func = EinsteinFunction::new(Span::call_site(), Vec::new());
    let mut index_use = IndexUse::new();
    parse_tensor_func(&mut input.into_iter(), &mut func, &mut index_use)?;
    Ok((func, index_use))
}
