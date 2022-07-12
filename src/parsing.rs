use crate::types::*;
use proc_macro2::{Delimiter, Group, Ident, Punct, Span, TokenStream, TokenTree};

fn generate_compilation_error(span: Span, message: &'static str) -> Result<(), TokenStream> {
    let invalid_stream = quote_spanned! {
        span => compile_error!(#message)
    };
    Err(invalid_stream)
}

fn parse_punct(
    punct: Punct,
    tokens: &mut impl Iterator<Item = TokenTree>,
    sequence: &mut EinsteinSequence,
    index_use: &mut IndexUse,
) -> Result<(), TokenStream> {
    let c = punct.as_char();
    if c == ';' {
        generate_compilation_error(punct.span(), "Character ';' is forbidden")?
    } else if c == '$' {
        let inverted_indexes = sequence.extract_previous_identifiers();
        let mut direct_indexes = inverted_indexes.clone();
        direct_indexes.reverse();
        for index_name in direct_indexes {
            let added = index_use.declare_new(index_name.clone());
            if !added {
                generate_compilation_error(index_name.span(), "Illegal reused index name")?
            }
        }
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
                    generate_compilation_error(group.span(), "Invalid content in indexes")?
                }
            }
            _ => generate_compilation_error(group.span(), "Invalid content in indexes")?,
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
            generate_compilation_error(group.span(), "Characters '{' and '}' are forbidden")?
        }
        Delimiter::Bracket => {
            if let Some(EinsteinAlternative::Tree(TokenTree::Ident(tensor_name))) =
                sequence.content.last()
            {
                let tensor_name = tensor_name.clone();
                let span = group.span();
                let indexes = parse_tensor_indexing(group)?;
                for (position, index_name) in indexes.iter().enumerate() {
                    let added = index_use.push(index_name.clone(), tensor_name.clone(), position);
                    if !added {
                        generate_compilation_error(index_name.span(), "Undeclared index")?
                    }
                }
                sequence.content.pop();
                sequence.content.push(EinsteinAlternative::TensorAccess {
                    tensor_name,
                    span,
                    indexes,
                });
            } else {
                generate_compilation_error(
                    group.span(),
                    "Invalid tensor name: an identifier was expected",
                )?
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
