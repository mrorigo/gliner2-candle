// Rust guideline compliant 2026-04-03
//! Whitespace tokenizer for GLiNER2.
//!
//! This module provides a fast regex-based tokenizer that matches the Python
//! `WhitespaceTokenSplitter` implementation. It handles URLs, emails, mentions,
//! hyphenated words, and other whitespace-separated tokens, mapping character
//! positions to token indices.

use regex::Regex;
use std::sync::LazyLock;

/// Regex pattern for whitespace tokenization.
///
/// Matches in order:
/// 1. URLs (http/https/www)
/// 2. Email addresses
/// 3. Mentions (@username)
/// 4. Words with optional hyphens/underscores
/// 5. Any non-whitespace character
static TOKEN_PATTERN: LazyLock<std::result::Result<Regex, regex::Error>> = LazyLock::new(|| {
    Regex::new(
        r"(?x)
        (?:https?://[^\s]+|www\.[^\s]+)
        |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
        |@[a-z0-9_]+
        |\w+(?:[-_]\w+)*
        |\S",
    )
});

/// A single token with its text and character position in the original text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    /// The token text.
    pub text: String,
    /// Start character index in the original text.
    pub start: usize,
    /// End character index in the original text (exclusive).
    pub end: usize,
}

impl Token {
    /// Create a new token.
    pub fn new(text: impl Into<String>, start: usize, end: usize) -> Self {
        Self {
            text: text.into(),
            start,
            end,
        }
    }

    /// Get the length of the token text.
    pub fn len(&self) -> usize {
        self.text.len()
    }

    /// Check if the token text is empty.
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}

/// Whitespace tokenizer that splits text into tokens using regex.
///
/// This tokenizer matches the Python `WhitespaceTokenSplitter` implementation
/// and handles:
/// - URLs (http, https, www)
/// - Email addresses
/// - Mentions (@username)
/// - Words with hyphens and underscores
/// - Individual non-whitespace characters
///
/// # Example
///
/// ```
/// use gliner2_candle::tokenizer::WhitespaceTokenizer;
///
/// let tokenizer = WhitespaceTokenizer::new();
/// let tokens = tokenizer.tokenize("Hello, world! Visit https://example.com");
///
/// assert_eq!(tokens[0].text, "hello");
/// assert_eq!(tokens[0].start, 0);
/// assert_eq!(tokens[0].end, 5);
/// ```
#[derive(Debug, Clone)]
pub struct WhitespaceTokenizer {
    /// Whether to lowercase tokens.
    lowercase: bool,
}

impl Default for WhitespaceTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl WhitespaceTokenizer {
    fn tokenize_with_regex(pattern: &Regex, text: &str) -> Vec<Token> {
        pattern
            .find_iter(text)
            .map(|m| Token {
                text: m.as_str().to_string(),
                start: m.start(),
                end: m.end(),
            })
            .collect()
    }

    fn tokenize_fallback(text: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut current_start: Option<usize> = None;

        for (idx, ch) in text.char_indices() {
            if ch.is_whitespace() {
                if let Some(start) = current_start.take() {
                    tokens.push(Token::new(&text[start..idx], start, idx));
                }
            } else if current_start.is_none() {
                current_start = Some(idx);
            }
        }

        if let Some(start) = current_start {
            tokens.push(Token::new(&text[start..], start, text.len()));
        }

        tokens
    }

    /// Create a new tokenizer with default settings (lowercase enabled).
    pub fn new() -> Self {
        Self { lowercase: true }
    }

    /// Create a new tokenizer with specified lowercase setting.
    pub fn with_lowercase(lowercase: bool) -> Self {
        Self { lowercase }
    }

    /// Tokenize text into a list of tokens.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to tokenize.
    ///
    /// # Returns
    ///
    /// A vector of tokens with their text and character positions.
    ///
    /// # Example
    ///
    /// ```
    /// use gliner2_candle::tokenizer::WhitespaceTokenizer;
    ///
    /// let tokenizer = WhitespaceTokenizer::new();
    /// let tokens = tokenizer.tokenize("Apple Inc.");
    /// assert_eq!(tokens.len(), 3);
    /// assert_eq!(tokens[0].text, "apple");
    /// assert_eq!(tokens[1].text, "inc");
    /// assert_eq!(tokens[2].text, ".");
    /// ```
    pub fn tokenize(&self, text: &str) -> Vec<Token> {
        let text = if self.lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        match TOKEN_PATTERN.as_ref() {
            Ok(pattern) => Self::tokenize_with_regex(pattern, &text),
            Err(err) => {
                tracing::error!(
                    error = %err,
                    "Failed to initialize tokenizer regex; using fallback splitter"
                );
                Self::tokenize_fallback(&text)
            }
        }
    }

    /// Tokenize text and return token texts only (without positions).
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to tokenize.
    ///
    /// # Returns
    ///
    /// A vector of token strings.
    pub fn tokenize_text(&self, text: &str) -> Vec<String> {
        self.tokenize(text).into_iter().map(|t| t.text).collect()
    }

    /// Build start and end mappings from tokens to character positions.
    ///
    /// These mappings are used to convert token indices back to character
    /// positions in the original text for span extraction.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The tokens to build mappings from.
    ///
    /// # Returns
    ///
    /// A tuple of (start_mapping, end_mapping) where:
    /// - `start_mapping[i]` is the character start position of token `i`
    /// - `end_mapping[i]` is the character end position of token `i`
    pub fn build_mappings(tokens: &[Token]) -> (Vec<usize>, Vec<usize>) {
        let start_mapping: Vec<usize> = tokens.iter().map(|t| t.start).collect();
        let end_mapping: Vec<usize> = tokens.iter().map(|t| t.end).collect();
        (start_mapping, end_mapping)
    }

    /// Tokenize text and return tokens with mappings in one call.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to tokenize.
    ///
    /// # Returns
    ///
    /// A tuple of (tokens, start_mapping, end_mapping).
    pub fn tokenize_with_mappings(&self, text: &str) -> (Vec<Token>, Vec<usize>, Vec<usize>) {
        let tokens = self.tokenize(text);
        let (start_mapping, end_mapping) = Self::build_mappings(&tokens);
        (tokens, start_mapping, end_mapping)
    }
}

/// Token mapping utilities for converting between token and character indices.
pub mod mapping {
    use super::*;

    /// Find the token index that contains a given character position.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The tokens to search.
    /// * `char_pos` - The character position to find.
    ///
    /// # Returns
    ///
    /// The token index, or `None` if not found.
    pub fn char_to_token(tokens: &[Token], char_pos: usize) -> Option<usize> {
        tokens
            .iter()
            .position(|t| char_pos >= t.start && char_pos < t.end)
    }

    /// Find the character range covered by a range of tokens.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The tokens to map.
    /// * `token_start` - Start token index (inclusive).
    /// * `token_end` - End token index (exclusive).
    ///
    /// # Returns
    ///
    /// A tuple of (char_start, char_end), or `None` if indices are invalid.
    pub fn token_to_char(
        tokens: &[Token],
        token_start: usize,
        token_end: usize,
    ) -> Option<(usize, usize)> {
        if token_start >= tokens.len() || token_end > tokens.len() || token_start >= token_end {
            return None;
        }
        Some((tokens[token_start].start, tokens[token_end - 1].end))
    }

    /// Extract text from original string using token indices.
    ///
    /// # Arguments
    ///
    /// * `text` - The original text.
    /// * `tokens` - The tokens.
    /// * `token_start` - Start token index (inclusive).
    /// * `token_end` - End token index (exclusive).
    ///
    /// # Returns
    ///
    /// The extracted text, or `None` if indices are invalid.
    pub fn extract_text_from_tokens(
        text: &str,
        tokens: &[Token],
        token_start: usize,
        token_end: usize,
    ) -> Option<String> {
        token_to_char(tokens, token_start, token_end)
            .map(|(start, end)| text[start..end].to_string())
    }
}

/// Pre-tokenized text with position mappings for efficient span extraction.
#[derive(Debug, Clone)]
pub struct TokenizedText {
    /// The original text.
    pub original_text: String,
    /// The tokens.
    pub tokens: Vec<Token>,
    /// Token texts only.
    pub token_texts: Vec<String>,
    /// Character start positions for each token.
    pub start_mapping: Vec<usize>,
    /// Character end positions for each token.
    pub end_mapping: Vec<usize>,
}

impl TokenizedText {
    /// Create a new `TokenizedText` from text using the given tokenizer.
    pub fn new(text: &str, tokenizer: &WhitespaceTokenizer) -> Self {
        let (tokens, start_mapping, end_mapping) = tokenizer.tokenize_with_mappings(text);
        let token_texts: Vec<String> = tokens.iter().map(|t| t.text.clone()).collect();

        Self {
            original_text: text.to_string(),
            tokens,
            token_texts,
            start_mapping,
            end_mapping,
        }
    }

    /// Get the number of tokens.
    pub fn num_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// Check if there are no tokens.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Extract text for a span of tokens.
    ///
    /// # Arguments
    ///
    /// * `start` - Start token index (inclusive).
    /// * `end` - End token index (exclusive).
    ///
    /// # Returns
    ///
    /// The extracted text, or empty string if indices are invalid.
    pub fn extract_span(&self, start: usize, end: usize) -> String {
        if start >= self.num_tokens() || end > self.num_tokens() || start >= end {
            return String::new();
        }
        let char_start = self.start_mapping[start];
        let char_end = self.end_mapping[end - 1];
        self.original_text[char_start..char_end].to_string()
    }

    /// Extract text for a span with confidence and position info.
    ///
    /// # Arguments
    ///
    /// * `start` - Start token index (inclusive).
    /// * `end` - End token index (exclusive).
    /// * `confidence` - Confidence score for the span.
    ///
    /// # Returns
    ///
    /// A tuple of (text, confidence, char_start, char_end).
    pub fn extract_span_with_info(
        &self,
        start: usize,
        end: usize,
        confidence: f32,
    ) -> Option<(String, f32, usize, usize)> {
        if start >= self.num_tokens() || end > self.num_tokens() || start >= end {
            return None;
        }
        let char_start = self.start_mapping[start];
        let char_end = self.end_mapping[end - 1];
        let text = self.original_text[char_start..char_end].to_string();
        if text.is_empty() {
            return None;
        }
        Some((text, confidence, char_start, char_end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenization() {
        let tokenizer = WhitespaceTokenizer::new();
        let tokens = tokenizer.tokenize("Hello, world!");

        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[0].start, 0);
        assert_eq!(tokens[0].end, 5);
        assert_eq!(tokens[1].text, ",");
        assert_eq!(tokens[2].text, "world");
        assert_eq!(tokens[3].text, "!");
    }

    #[test]
    fn test_url_tokenization() {
        let tokenizer = WhitespaceTokenizer::new();
        let tokens = tokenizer.tokenize("Visit https://example.com today");

        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "visit");
        assert_eq!(tokens[1].text, "https://example.com");
        assert_eq!(tokens[2].text, "today");
    }

    #[test]
    fn test_email_tokenization() {
        let tokenizer = WhitespaceTokenizer::new();
        let tokens = tokenizer.tokenize("Contact user@example.com please");

        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "contact");
        assert_eq!(tokens[1].text, "user@example.com");
        assert_eq!(tokens[2].text, "please");
    }

    #[test]
    fn test_mention_tokenization() {
        let tokenizer = WhitespaceTokenizer::new();
        let tokens = tokenizer.tokenize("Follow @username now");

        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "follow");
        assert_eq!(tokens[1].text, "@username");
        assert_eq!(tokens[2].text, "now");
    }

    #[test]
    fn test_hyphenated_words() {
        let tokenizer = WhitespaceTokenizer::new();
        let tokens = tokenizer.tokenize("state-of-the-art");

        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "state-of-the-art");
    }

    #[test]
    fn test_underscored_words() {
        let tokenizer = WhitespaceTokenizer::new();
        let tokens = tokenizer.tokenize("my_variable_name");

        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "my_variable_name");
    }

    #[test]
    fn test_no_lowercase() {
        let tokenizer = WhitespaceTokenizer::with_lowercase(false);
        let tokens = tokenizer.tokenize("Hello World");

        assert_eq!(tokens[0].text, "Hello");
        assert_eq!(tokens[1].text, "World");
    }

    #[test]
    fn test_empty_text() {
        let tokenizer = WhitespaceTokenizer::new();
        let tokens = tokenizer.tokenize("");

        assert!(tokens.is_empty());
    }

    #[test]
    fn test_whitespace_only() {
        let tokenizer = WhitespaceTokenizer::new();
        let tokens = tokenizer.tokenize("   \t\n  ");

        assert!(tokens.is_empty());
    }

    #[test]
    fn test_mappings() {
        let tokenizer = WhitespaceTokenizer::new();
        let tokens = tokenizer.tokenize("Apple Inc.");
        let (start_mapping, end_mapping) = WhitespaceTokenizer::build_mappings(&tokens);

        assert_eq!(start_mapping, vec![0, 6, 9]);
        assert_eq!(end_mapping, vec![5, 9, 10]);
    }

    #[test]
    fn test_tokenized_text() {
        let tokenizer = WhitespaceTokenizer::new();
        let tokenized = TokenizedText::new("Apple Inc.", &tokenizer);

        assert_eq!(tokenized.num_tokens(), 3);
        assert_eq!(tokenized.token_texts, vec!["apple", "inc", "."]);
        assert_eq!(tokenized.extract_span(0, 1), "Apple");
        assert_eq!(tokenized.extract_span(0, 2), "Apple Inc");
    }

    #[test]
    fn test_char_to_token() {
        let tokenizer = WhitespaceTokenizer::new();
        let tokens = tokenizer.tokenize("Hello world");

        assert_eq!(mapping::char_to_token(&tokens, 0), Some(0));
        assert_eq!(mapping::char_to_token(&tokens, 3), Some(0));
        assert_eq!(mapping::char_to_token(&tokens, 6), Some(1));
        assert_eq!(mapping::char_to_token(&tokens, 100), None);
    }

    #[test]
    fn test_token_to_char() {
        let tokenizer = WhitespaceTokenizer::new();
        let tokens = tokenizer.tokenize("Hello world");

        assert_eq!(mapping::token_to_char(&tokens, 0, 1), Some((0, 5)));
        assert_eq!(mapping::token_to_char(&tokens, 0, 2), Some((0, 11)));
        assert_eq!(mapping::token_to_char(&tokens, 1, 2), Some((6, 11)));
        assert_eq!(mapping::token_to_char(&tokens, 0, 0), None);
    }

    #[test]
    fn test_complex_text() {
        let tokenizer = WhitespaceTokenizer::new();
        let text = "Apple CEO Tim Cook announced iPhone 15 in Cupertino.";
        let tokens = tokenizer.tokenize(text);

        assert_eq!(tokens.len(), 10);
        assert_eq!(tokens[0].text, "apple");
        assert_eq!(tokens[1].text, "ceo");
        assert_eq!(tokens[2].text, "tim");
        assert_eq!(tokens[3].text, "cook");
        assert_eq!(tokens[4].text, "announced");
        assert_eq!(tokens[5].text, "iphone");
        assert_eq!(tokens[6].text, "15");
        assert_eq!(tokens[7].text, "in");
        assert_eq!(tokens[8].text, "cupertino");
        assert_eq!(tokens[9].text, ".");
    }

    #[test]
    fn test_span_extraction_with_info() {
        let tokenizer = WhitespaceTokenizer::new();
        let tokenized = TokenizedText::new("Apple Inc. is great.", &tokenizer);

        let info = tokenized.extract_span_with_info(0, 2, 0.95);
        assert!(info.is_some());
        let (text, conf, start, end) = info.unwrap();
        assert_eq!(text, "Apple Inc");
        assert!((conf - 0.95).abs() < f32::EPSILON);
        assert_eq!(start, 0);
        assert_eq!(end, 9);
    }
}
