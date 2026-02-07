use tantivy::tokenizer::{BoxTokenStream, Token, TokenStream, Tokenizer};

/// Tokenizer tuned for code identifiers.
///
/// Splits on punctuation/separators and also on camelCase boundaries.
#[derive(Clone, Default)]
pub struct CodeTokenizer;

#[derive(Clone)]
struct CodeTokenStream {
    tokens: Vec<Token>,
    current: Token,
    index: usize,
}

impl Tokenizer for CodeTokenizer {
    type TokenStream<'a> = BoxTokenStream<'a>;

    fn token_stream<'a>(&mut self, text: &'a str) -> Self::TokenStream<'a> {
        let tokens = tokenize_code(text);
        BoxTokenStream::new(CodeTokenStream {
            tokens,
            current: Token::default(),
            index: 0,
        })
    }
}

impl TokenStream for CodeTokenStream {
    fn advance(&mut self) -> bool {
        if let Some(token) = self.tokens.get(self.index).cloned() {
            self.current = token;
            self.index += 1;
            true
        } else {
            false
        }
    }

    fn token(&self) -> &Token {
        &self.current
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.current
    }
}

fn tokenize_code(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut position = 0usize;

    let mut iter = text.char_indices().peekable();
    while let Some((start, ch)) = iter.peek().copied() {
        if !is_identifier_char(ch) {
            iter.next();
            continue;
        }

        iter.next();
        let mut end = start + ch.len_utf8();
        while let Some((idx, next_ch)) = iter.peek().copied() {
            if !is_identifier_char(next_ch) {
                end = idx;
                break;
            }
            iter.next();
            end = idx + next_ch.len_utf8();
        }

        for (seg_start, seg_end) in split_identifier(text, start, end) {
            if seg_start >= seg_end {
                continue;
            }
            let term = text[seg_start..seg_end].to_ascii_lowercase();
            if term.is_empty() {
                continue;
            }

            tokens.push(Token {
                offset_from: seg_start,
                offset_to: seg_end,
                position,
                text: term,
                position_length: 1,
            });
            position += 1;
        }
    }

    tokens
}

fn split_identifier(text: &str, start: usize, end: usize) -> Vec<(usize, usize)> {
    // First split by snake/kebab separators.
    let mut coarse = Vec::new();
    let mut segment_start = start;
    for (idx, ch) in text[start..end].char_indices() {
        if ch == '_' || ch == '-' {
            let abs = start + idx;
            if segment_start < abs {
                coarse.push((segment_start, abs));
            }
            segment_start = abs + ch.len_utf8();
        }
    }
    if segment_start < end {
        coarse.push((segment_start, end));
    }

    let mut fine = Vec::new();
    for (coarse_start, coarse_end) in coarse {
        fine.extend(split_camel_case(text, coarse_start, coarse_end));
    }
    fine
}

fn split_camel_case(text: &str, start: usize, end: usize) -> Vec<(usize, usize)> {
    let mut pieces = Vec::new();
    let mut boundary = start;

    let mut chars: Vec<(usize, char)> = text[start..end]
        .char_indices()
        .map(|(i, c)| (start + i, c))
        .collect();
    chars.push((end, '\0'));

    for i in 1..chars.len().saturating_sub(1) {
        let (_prev_idx, prev) = chars[i - 1];
        let (cur_idx, cur) = chars[i];
        let (_next_idx, next) = chars[i + 1];

        let lower_to_upper =
            (prev.is_ascii_lowercase() || prev.is_ascii_digit()) && cur.is_ascii_uppercase();
        let acronym_to_word =
            prev.is_ascii_uppercase() && cur.is_ascii_uppercase() && next.is_ascii_lowercase();

        if lower_to_upper || acronym_to_word {
            if boundary < cur_idx {
                pieces.push((boundary, cur_idx));
            }
            boundary = cur_idx;
        }
    }

    if boundary < end {
        pieces.push((boundary, end));
    }

    pieces
}

fn is_identifier_char(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '_' || ch == '-'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_snake_and_camel() {
        let mut tokenizer = CodeTokenizer;
        let mut stream = tokenizer.token_stream("UserConfig process_data HTTPServer");
        let mut terms = Vec::new();
        while stream.advance() {
            terms.push(stream.token().text.clone());
        }

        assert_eq!(
            terms,
            vec!["user", "config", "process", "data", "http", "server"]
        );
    }
}
