use proc_macro::TokenStream;

/// Derive macro for Table types. Generates Bundle impl, Table trait impl,
/// and typed row reference structs (FooRef, FooMut).
#[proc_macro_derive(Table)]
pub fn derive_table(input: TokenStream) -> TokenStream {
    let _ = input;
    TokenStream::new() // stub — generates nothing yet
}
