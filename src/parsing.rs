use crate::types::*;
use proc_macro2::{Delimiter, Group, Ident, Punct, TokenStream, TokenTree};

fn parse_punct(
    punct: Punct,
    tokens: &mut impl Iterator<Item = TokenTree>,
    sequence: &mut EinsteinSequence,
    index_use: &mut IndexUse,
) -> Result<(), TokenStream> {
    let c = punct.as_char();
    if c == ';' {
        let invalid_stream = quote_spanned! {
            punct.span() => compile_error!("Character ';' is forbidden")
        };
        return Err(invalid_stream);
    } else if c == '$' {
        let inverted_indexes = sequence.extract_previous_identifiers();
        let mut new_sequence = EinsteinSequence::naked(punct.span());
        parse_sequence(tokens, &mut new_sequence, index_use)?;
        let func = EinsteinFunction::new(inverted_indexes, new_sequence);
        sequence.content.push(EinsteinAlternative::Func(func));
    } else {
        sequence.push_token(TokenTree::Punct(punct.clone()));
    }
    Ok(())
}

fn parse_tensor_indexing(group: Group) -> Result<Vec<Ident>, TokenStream> {
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
    group: Group,
    sequence: &mut EinsteinSequence,
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
                sequence.content.last()
            {
                let tensor_name = tensor_name.clone();
                let span = group.span();
                let indexes = parse_tensor_indexing(group)?;
                for (position, index_name) in indexes.iter().enumerate() {
                    index_use.push(index_name.clone(), tensor_name.clone(), position)
                }
                sequence.content.pop();
                sequence.content.push(EinsteinAlternative::TensorAccess {
                    tensor_name,
                    span,
                    indexes,
                });
            } else {
                let invalid_stream = quote_spanned! {
                    group.span() => compile_error!("Invalid tensor name: an identifier was expected")
                };
                return Err(invalid_stream);
            }
        }
        delimiter => {
            let mut new_sequence = if delimiter == Delimiter::Parenthesis {
                EinsteinSequence::with_parens(group.span())
            } else {
                EinsteinSequence::naked(group.span())
            };
            parse_sequence(
                &mut group.stream().into_iter(),
                &mut new_sequence,
                index_use,
            )?;
            sequence
                .content
                .push(EinsteinAlternative::Seq(new_sequence));
        }
    }
    Ok(())
}

fn parse_sequence(
    tokens: &mut impl Iterator<Item = TokenTree>,
    sequence: &mut EinsteinSequence,
    index_use: &mut IndexUse,
) -> Result<(), TokenStream> {
    while let Some(token) = tokens.next() {
        match token {
            TokenTree::Punct(punct) => parse_punct(punct, tokens, sequence, index_use)?,
            TokenTree::Group(group) => parse_group(group, sequence, index_use)?,
            _ => sequence.push_token(token),
        }
    }
    Ok(())
}

pub fn parse(input: proc_macro::TokenStream) -> Result<(EinsteinSequence, IndexUse), TokenStream> {
    let input: TokenStream = input.into();
    let mut sequence = EinsteinSequence::initial();
    let mut index_use = IndexUse::new();
    parse_sequence(&mut input.into_iter(), &mut sequence, &mut index_use)?;
    Ok((sequence, index_use))
}
