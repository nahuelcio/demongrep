use crate::file::Language;
use super::ChunkKind;
use tree_sitter::Node;

/// Language-specific code extraction logic
///
/// Each language has different AST node types and conventions for:
/// - Finding definitions (functions, classes, etc.)
/// - Extracting names
/// - Building signatures
/// - Finding docstrings
///
/// This trait allows us to handle multiple languages with proper semantics.
pub trait LanguageExtractor: Send + Sync {
    /// Get the AST node types that represent definitions in this language
    ///
    /// For example:
    /// - Rust: `["function_item", "struct_item", "impl_item", ...]`
    /// - Python: `["function_definition", "class_definition"]`
    fn definition_types(&self) -> &[&'static str];

    /// Extract the name from a definition node
    ///
    /// Returns None if the node has no name (anonymous)
    fn extract_name(&self, node: Node, source: &[u8]) -> Option<String>;

    /// Extract a function/method signature
    ///
    /// Examples:
    /// - Rust: `fn sort<T: Ord>(items: Vec<T>) -> Vec<T>`
    /// - Python: `def process(data: List[str]) -> Dict[str, int]`
    /// - TypeScript: `function compute(x: number): string`
    fn extract_signature(&self, node: Node, source: &[u8]) -> Option<String>;

    /// Extract docstring/documentation comments
    ///
    /// Different languages have different conventions:
    /// - Rust: `/// ` and `/** */`
    /// - Python: First string literal in function/class body
    /// - JavaScript/TypeScript: JSDoc `/** */`
    fn extract_docstring(&self, node: Node, source: &[u8]) -> Option<String>;

    /// Classify a node into a ChunkKind
    fn classify(&self, node: Node) -> ChunkKind;

    /// Check if a node is a definition
    fn is_definition(&self, node: Node) -> bool {
        self.definition_types().contains(&node.kind())
    }

    /// Build a label for a node (e.g., "Function: foo", "Class: Bar")
    fn build_label(&self, node: Node, source: &[u8]) -> Option<String> {
        let name = self.extract_name(node, source)?;
        let kind = self.classify(node);

        Some(match kind {
            ChunkKind::Function => format!("Function: {}", name),
            ChunkKind::Method => format!("Method: {}", name),
            ChunkKind::Class => format!("Class: {}", name),
            ChunkKind::Struct => format!("Struct: {}", name),
            ChunkKind::Enum => format!("Enum: {}", name),
            ChunkKind::Trait => format!("Trait: {}", name),
            ChunkKind::Interface => format!("Interface: {}", name),
            ChunkKind::Impl => format!("Impl: {}", name),
            ChunkKind::Mod => format!("Module: {}", name),
            ChunkKind::TypeAlias => format!("Type: {}", name),
            ChunkKind::Const => format!("Const: {}", name),
            ChunkKind::Static => format!("Static: {}", name),
            _ => format!("Symbol: {}", name),
        })
    }
}

/// Get the appropriate extractor for a language
pub fn get_extractor(language: Language) -> Option<Box<dyn LanguageExtractor>> {
    match language {
        Language::Rust => Some(Box::new(RustExtractor)),
        Language::Python => Some(Box::new(PythonExtractor)),
        Language::JavaScript | Language::TypeScript => Some(Box::new(TypeScriptExtractor)),
        Language::CSharp => Some(Box::new(CSharpExtractor)),
        Language::Go => Some(Box::new(GoExtractor)),
        Language::Java => Some(Box::new(JavaExtractor)),
        Language::C | Language::Cpp => Some(Box::new(CppExtractor)),
        Language::Ruby => Some(Box::new(RubyExtractor)),
        Language::Php => Some(Box::new(PhpExtractor)),
        Language::Shell => Some(Box::new(BashExtractor)),
        _ => None,
    }
}

/// Rust language extractor
pub struct RustExtractor;

impl LanguageExtractor for RustExtractor {
    fn definition_types(&self) -> &[&'static str] {
        &[
            "function_item",
            "struct_item",
            "enum_item",
            "impl_item",
            "trait_item",
            "type_item",
            "mod_item",
            "const_item",
            "static_item",
        ]
    }

    fn extract_name(&self, node: Node, source: &[u8]) -> Option<String> {
        // Rust has consistent "name" field for most definitions
        node.child_by_field_name("name")?
            .utf8_text(source)
            .ok()
            .map(String::from)
    }

    fn extract_signature(&self, node: Node, source: &[u8]) -> Option<String> {
        match node.kind() {
            "function_item" => {
                // Build: fn name<T>(params) -> Return
                let mut sig = String::from("fn ");

                // Add name
                if let Some(name) = node.child_by_field_name("name") {
                    if let Ok(name_text) = name.utf8_text(source) {
                        sig.push_str(name_text);
                    }
                }

                // Add type parameters
                if let Some(type_params) = node.child_by_field_name("type_parameters") {
                    if let Ok(params_text) = type_params.utf8_text(source) {
                        sig.push_str(params_text);
                    }
                }

                // Add parameters
                if let Some(params) = node.child_by_field_name("parameters") {
                    if let Ok(params_text) = params.utf8_text(source) {
                        sig.push_str(params_text);
                    }
                }

                // Add return type
                if let Some(return_type) = node.child_by_field_name("return_type") {
                    if let Ok(ret_text) = return_type.utf8_text(source) {
                        sig.push_str(" -> ");
                        sig.push_str(ret_text);
                    }
                }

                Some(sig)
            }
            "struct_item" => {
                // Build: struct Name<T>
                let mut sig = String::from("struct ");

                if let Some(name) = node.child_by_field_name("name") {
                    if let Ok(name_text) = name.utf8_text(source) {
                        sig.push_str(name_text);
                    }
                }

                if let Some(type_params) = node.child_by_field_name("type_parameters") {
                    if let Ok(params_text) = type_params.utf8_text(source) {
                        sig.push_str(params_text);
                    }
                }

                Some(sig)
            }
            "enum_item" => {
                // Build: enum Name<T>
                let mut sig = String::from("enum ");

                if let Some(name) = node.child_by_field_name("name") {
                    if let Ok(name_text) = name.utf8_text(source) {
                        sig.push_str(name_text);
                    }
                }

                if let Some(type_params) = node.child_by_field_name("type_parameters") {
                    if let Ok(params_text) = type_params.utf8_text(source) {
                        sig.push_str(params_text);
                    }
                }

                Some(sig)
            }
            "trait_item" => {
                // Build: trait Name<T>
                let mut sig = String::from("trait ");

                if let Some(name) = node.child_by_field_name("name") {
                    if let Ok(name_text) = name.utf8_text(source) {
                        sig.push_str(name_text);
                    }
                }

                if let Some(type_params) = node.child_by_field_name("type_parameters") {
                    if let Ok(params_text) = type_params.utf8_text(source) {
                        sig.push_str(params_text);
                    }
                }

                Some(sig)
            }
            "impl_item" => {
                // Build: impl<T> Trait for Type
                let mut sig = String::from("impl");

                if let Some(type_params) = node.child_by_field_name("type_parameters") {
                    if let Ok(params_text) = type_params.utf8_text(source) {
                        sig.push_str(params_text);
                    }
                }

                if let Some(trait_name) = node.child_by_field_name("trait") {
                    if let Ok(trait_text) = trait_name.utf8_text(source) {
                        sig.push(' ');
                        sig.push_str(trait_text);
                        sig.push_str(" for");
                    }
                }

                if let Some(type_name) = node.child_by_field_name("type") {
                    if let Ok(type_text) = type_name.utf8_text(source) {
                        sig.push(' ');
                        sig.push_str(type_text);
                    }
                }

                Some(sig)
            }
            _ => None,
        }
    }

    fn extract_docstring(&self, node: Node, source: &[u8]) -> Option<String> {
        // Look for line_comment or block_comment nodes immediately before this node
        // Tree-sitter includes them as named siblings in some grammars

        // For now, we'll look at the previous siblings
        let parent = node.parent()?;
        let node_index = (0..parent.named_child_count())
            .find(|&i| parent.named_child(i).map(|c| c.id()) == Some(node.id()))?;

        if node_index > 0 {
            if let Some(prev) = parent.named_child(node_index - 1) {
                if prev.kind() == "line_comment" || prev.kind() == "block_comment" {
                    if let Ok(text) = prev.utf8_text(source) {
                        // Check if it's a doc comment (/// or /**)
                        if text.trim_start().starts_with("///") ||
                           text.trim_start().starts_with("/**") {
                            return Some(text.to_string());
                        }
                    }
                }
            }
        }

        None
    }

    fn classify(&self, node: Node) -> ChunkKind {
        match node.kind() {
            "function_item" => {
                // Check if it's a method (inside impl block)
                if let Some(parent) = node.parent() {
                    if parent.kind() == "declaration_list" {
                        if let Some(grandparent) = parent.parent() {
                            if grandparent.kind() == "impl_item" {
                                return ChunkKind::Method;
                            }
                        }
                    }
                }
                ChunkKind::Function
            }
            "struct_item" => ChunkKind::Struct,
            "enum_item" => ChunkKind::Enum,
            "impl_item" => ChunkKind::Impl,
            "trait_item" => ChunkKind::Trait,
            "type_item" => ChunkKind::TypeAlias,
            "mod_item" => ChunkKind::Mod,
            "const_item" => ChunkKind::Const,
            "static_item" => ChunkKind::Static,
            _ => ChunkKind::Other,
        }
    }
}

/// Python language extractor
pub struct PythonExtractor;

impl LanguageExtractor for PythonExtractor {
    fn definition_types(&self) -> &[&'static str] {
        &["function_definition", "class_definition"]
    }

    fn extract_name(&self, node: Node, source: &[u8]) -> Option<String> {
        node.child_by_field_name("name")?
            .utf8_text(source)
            .ok()
            .map(String::from)
    }

    fn extract_signature(&self, node: Node, source: &[u8]) -> Option<String> {
        match node.kind() {
            "function_definition" => {
                // Build: def name(params) -> Return:
                let mut sig = String::from("def ");

                if let Some(name) = node.child_by_field_name("name") {
                    if let Ok(name_text) = name.utf8_text(source) {
                        sig.push_str(name_text);
                    }
                }

                if let Some(params) = node.child_by_field_name("parameters") {
                    if let Ok(params_text) = params.utf8_text(source) {
                        sig.push_str(params_text);
                    }
                }

                if let Some(return_type) = node.child_by_field_name("return_type") {
                    if let Ok(ret_text) = return_type.utf8_text(source) {
                        sig.push_str(" -> ");
                        sig.push_str(ret_text);
                    }
                }

                Some(sig)
            }
            "class_definition" => {
                // Build: class Name(Base):
                let mut sig = String::from("class ");

                if let Some(name) = node.child_by_field_name("name") {
                    if let Ok(name_text) = name.utf8_text(source) {
                        sig.push_str(name_text);
                    }
                }

                if let Some(superclasses) = node.child_by_field_name("superclasses") {
                    if let Ok(bases_text) = superclasses.utf8_text(source) {
                        sig.push_str(bases_text);
                    }
                }

                Some(sig)
            }
            _ => None,
        }
    }

    fn extract_docstring(&self, node: Node, source: &[u8]) -> Option<String> {
        // Python docstrings are the first statement in the body if it's a string
        let body = node.child_by_field_name("body")?;

        let mut cursor = body.walk();
        for child in body.named_children(&mut cursor) {
            if child.kind() == "expression_statement" {
                // Check if it contains a string
                let mut expr_cursor = child.walk();
                for expr_child in child.named_children(&mut expr_cursor) {
                    if expr_child.kind() == "string" {
                        return expr_child.utf8_text(source).ok().map(String::from);
                    }
                }
            }
            // Only check first statement
            break;
        }

        None
    }

    fn classify(&self, node: Node) -> ChunkKind {
        match node.kind() {
            "function_definition" => {
                // Check if it's a method (inside class)
                if let Some(parent) = node.parent() {
                    if parent.kind() == "block" {
                        if let Some(grandparent) = parent.parent() {
                            if grandparent.kind() == "class_definition" {
                                return ChunkKind::Method;
                            }
                        }
                    }
                }
                ChunkKind::Function
            }
            "class_definition" => ChunkKind::Class,
            _ => ChunkKind::Other,
        }
    }
}

/// TypeScript/JavaScript language extractor
pub struct TypeScriptExtractor;

impl LanguageExtractor for TypeScriptExtractor {
    fn definition_types(&self) -> &[&'static str] {
        &[
            "function_declaration",
            "function",
            "method_definition",
            "class_declaration",
            "class",
            "interface_declaration",
            "type_alias_declaration",
            "enum_declaration",
            // Arrow functions assigned to const
            "lexical_declaration",
            "variable_declaration",
        ]
    }

    fn extract_name(&self, node: Node, source: &[u8]) -> Option<String> {
        // Try name field first
        if let Some(name) = node.child_by_field_name("name") {
            if let Ok(text) = name.utf8_text(source) {
                return Some(text.to_string());
            }
        }

        // For variable declarations, look for identifier
        if node.kind() == "lexical_declaration" || node.kind() == "variable_declaration" {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                if child.kind() == "variable_declarator" {
                    if let Some(name) = child.child_by_field_name("name") {
                        if let Ok(text) = name.utf8_text(source) {
                            return Some(text.to_string());
                        }
                    }
                }
            }
        }

        None
    }

    fn extract_signature(&self, node: Node, source: &[u8]) -> Option<String> {
        match node.kind() {
            "function_declaration" | "function" => {
                let mut sig = String::from("function ");

                if let Some(name) = node.child_by_field_name("name") {
                    if let Ok(name_text) = name.utf8_text(source) {
                        sig.push_str(name_text);
                    }
                }

                if let Some(params) = node.child_by_field_name("parameters") {
                    if let Ok(params_text) = params.utf8_text(source) {
                        sig.push_str(params_text);
                    }
                }

                if let Some(return_type) = node.child_by_field_name("return_type") {
                    if let Ok(ret_text) = return_type.utf8_text(source) {
                        sig.push_str(": ");
                        sig.push_str(ret_text);
                    }
                }

                Some(sig)
            }
            "class_declaration" | "class" => {
                let mut sig = String::from("class ");

                if let Some(name) = node.child_by_field_name("name") {
                    if let Ok(name_text) = name.utf8_text(source) {
                        sig.push_str(name_text);
                    }
                }

                Some(sig)
            }
            _ => None,
        }
    }

    fn extract_docstring(&self, node: Node, source: &[u8]) -> Option<String> {
        // Look for JSDoc comments (/** */) before the node
        // Similar to Rust approach
        let parent = node.parent()?;
        let node_index = (0..parent.named_child_count())
            .find(|&i| parent.named_child(i).map(|c| c.id()) == Some(node.id()))?;

        if node_index > 0 {
            if let Some(prev) = parent.named_child(node_index - 1) {
                if prev.kind() == "comment" {
                    if let Ok(text) = prev.utf8_text(source) {
                        if text.trim_start().starts_with("/**") {
                            return Some(text.to_string());
                        }
                    }
                }
            }
        }

        None
    }

    fn classify(&self, node: Node) -> ChunkKind {
        match node.kind() {
            "function_declaration" | "function" => ChunkKind::Function,
            "method_definition" => ChunkKind::Method,
            "class_declaration" | "class" => ChunkKind::Class,
            "interface_declaration" => ChunkKind::Interface,
            "type_alias_declaration" => ChunkKind::TypeAlias,
            "enum_declaration" => ChunkKind::Enum,
            "lexical_declaration" | "variable_declaration" => {
                // Check if it's an arrow function
                // If so, treat as Function, otherwise Other
                ChunkKind::Function
            }
            _ => ChunkKind::Other,
        }
    }
}

/// C# language extractor
pub struct CSharpExtractor;

impl LanguageExtractor for CSharpExtractor {
    fn definition_types(&self) -> &[&'static str] {
        &[
            "method_declaration",
            "class_declaration", 
            "struct_declaration",
            "interface_declaration",
            "enum_declaration",
            "record_declaration",
            "property_declaration",
            "constructor_declaration",
            "namespace_declaration",
            "delegate_declaration",
        ]
    }

    fn extract_name(&self, node: Node, source: &[u8]) -> Option<String> {
        node.child_by_field_name("name")?
            .utf8_text(source)
            .ok()
            .map(String::from)
    }

    fn extract_signature(&self, node: Node, source: &[u8]) -> Option<String> {
        match node.kind() {
            "method_declaration" => {
                let mut sig = String::new();
                
                // Return type
                if let Some(ret) = node.child_by_field_name("type") {
                    if let Ok(text) = ret.utf8_text(source) {
                        sig.push_str(text);
                        sig.push(' ');
                    }
                }
                
                // Name
                if let Some(name) = node.child_by_field_name("name") {
                    if let Ok(text) = name.utf8_text(source) {
                        sig.push_str(text);
                    }
                }
                
                // Type parameters
                if let Some(type_params) = node.child_by_field_name("type_parameters") {
                    if let Ok(text) = type_params.utf8_text(source) {
                        sig.push_str(text);
                    }
                }
                
                // Parameters
                if let Some(params) = node.child_by_field_name("parameters") {
                    if let Ok(text) = params.utf8_text(source) {
                        sig.push_str(text);
                    }
                }
                
                Some(sig)
            }
            "class_declaration" | "struct_declaration" | "interface_declaration" => {
                let keyword = match node.kind() {
                    "class_declaration" => "class",
                    "struct_declaration" => "struct", 
                    "interface_declaration" => "interface",
                    _ => return None,
                };
                
                let mut sig = String::from(keyword);
                sig.push(' ');
                
                if let Some(name) = node.child_by_field_name("name") {
                    if let Ok(text) = name.utf8_text(source) {
                        sig.push_str(text);
                    }
                }
                
                if let Some(type_params) = node.child_by_field_name("type_parameters") {
                    if let Ok(text) = type_params.utf8_text(source) {
                        sig.push_str(text);
                    }
                }
                
                Some(sig)
            }
            _ => None,
        }
    }

    fn extract_docstring(&self, node: Node, source: &[u8]) -> Option<String> {
        // Look for XML doc comments (/// or /** */)
        let parent = node.parent()?;
        let node_index = (0..parent.named_child_count())
            .find(|&i| parent.named_child(i).map(|c| c.id()) == Some(node.id()))?;

        if node_index > 0 {
            if let Some(prev) = parent.named_child(node_index - 1) {
                if prev.kind() == "comment" {
                    if let Ok(text) = prev.utf8_text(source) {
                        if text.trim_start().starts_with("///") || 
                           text.trim_start().starts_with("/**") {
                            return Some(text.to_string());
                        }
                    }
                }
            }
        }
        None
    }

    fn classify(&self, node: Node) -> ChunkKind {
        match node.kind() {
            "method_declaration" | "constructor_declaration" => {
                // Check if inside a class/struct/interface
                if let Some(parent) = node.parent() {
                    if parent.kind() == "declaration_list" {
                        return ChunkKind::Method;
                    }
                }
                ChunkKind::Function
            }
            "class_declaration" | "record_declaration" => ChunkKind::Class,
            "struct_declaration" => ChunkKind::Struct,
            "interface_declaration" => ChunkKind::Interface,
            "enum_declaration" => ChunkKind::Enum,
            "namespace_declaration" => ChunkKind::Mod,
            "property_declaration" => ChunkKind::Other,
            "delegate_declaration" => ChunkKind::TypeAlias,
            _ => ChunkKind::Other,
        }
    }
}

/// Go language extractor
pub struct GoExtractor;

impl LanguageExtractor for GoExtractor {
    fn definition_types(&self) -> &[&'static str] {
        &[
            "function_declaration",
            "method_declaration",
            "type_declaration",
            "type_spec",
        ]
    }

    fn extract_name(&self, node: Node, source: &[u8]) -> Option<String> {
        node.child_by_field_name("name")
            .or_else(|| {
                // For type_spec, name might be nested
                if node.kind() == "type_spec" {
                    node.child_by_field_name("name")
                } else {
                    None
                }
            })?
            .utf8_text(source)
            .ok()
            .map(String::from)
    }

    fn extract_signature(&self, node: Node, source: &[u8]) -> Option<String> {
        match node.kind() {
            "function_declaration" => {
                let mut sig = String::from("func ");
                
                if let Some(name) = node.child_by_field_name("name") {
                    if let Ok(text) = name.utf8_text(source) {
                        sig.push_str(text);
                    }
                }
                
                if let Some(params) = node.child_by_field_name("parameters") {
                    if let Ok(text) = params.utf8_text(source) {
                        sig.push_str(text);
                    }
                }
                
                if let Some(result) = node.child_by_field_name("result") {
                    if let Ok(text) = result.utf8_text(source) {
                        sig.push(' ');
                        sig.push_str(text);
                    }
                }
                
                Some(sig)
            }
            "method_declaration" => {
                let mut sig = String::from("func ");
                
                if let Some(receiver) = node.child_by_field_name("receiver") {
                    if let Ok(text) = receiver.utf8_text(source) {
                        sig.push_str(text);
                        sig.push(' ');
                    }
                }
                
                if let Some(name) = node.child_by_field_name("name") {
                    if let Ok(text) = name.utf8_text(source) {
                        sig.push_str(text);
                    }
                }
                
                if let Some(params) = node.child_by_field_name("parameters") {
                    if let Ok(text) = params.utf8_text(source) {
                        sig.push_str(text);
                    }
                }
                
                Some(sig)
            }
            "type_spec" => {
                let mut sig = String::from("type ");
                
                if let Some(name) = node.child_by_field_name("name") {
                    if let Ok(text) = name.utf8_text(source) {
                        sig.push_str(text);
                    }
                }
                
                Some(sig)
            }
            _ => None,
        }
    }

    fn extract_docstring(&self, node: Node, source: &[u8]) -> Option<String> {
        let parent = node.parent()?;
        let node_index = (0..parent.named_child_count())
            .find(|&i| parent.named_child(i).map(|c| c.id()) == Some(node.id()))?;

        if node_index > 0 {
            if let Some(prev) = parent.named_child(node_index - 1) {
                if prev.kind() == "comment" {
                    return prev.utf8_text(source).ok().map(String::from);
                }
            }
        }
        None
    }

    fn classify(&self, node: Node) -> ChunkKind {
        match node.kind() {
            "function_declaration" => ChunkKind::Function,
            "method_declaration" => ChunkKind::Method,
            "type_declaration" | "type_spec" => ChunkKind::Struct,
            _ => ChunkKind::Other,
        }
    }
}

/// Java language extractor
pub struct JavaExtractor;

impl LanguageExtractor for JavaExtractor {
    fn definition_types(&self) -> &[&'static str] {
        &[
            "method_declaration",
            "constructor_declaration",
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "annotation_type_declaration",
            "record_declaration",
        ]
    }

    fn extract_name(&self, node: Node, source: &[u8]) -> Option<String> {
        node.child_by_field_name("name")?
            .utf8_text(source)
            .ok()
            .map(String::from)
    }

    fn extract_signature(&self, node: Node, source: &[u8]) -> Option<String> {
        match node.kind() {
            "method_declaration" => {
                let mut sig = String::new();
                
                if let Some(ret) = node.child_by_field_name("type") {
                    if let Ok(text) = ret.utf8_text(source) {
                        sig.push_str(text);
                        sig.push(' ');
                    }
                }
                
                if let Some(name) = node.child_by_field_name("name") {
                    if let Ok(text) = name.utf8_text(source) {
                        sig.push_str(text);
                    }
                }
                
                if let Some(params) = node.child_by_field_name("parameters") {
                    if let Ok(text) = params.utf8_text(source) {
                        sig.push_str(text);
                    }
                }
                
                Some(sig)
            }
            "class_declaration" | "interface_declaration" | "enum_declaration" => {
                let keyword = match node.kind() {
                    "class_declaration" => "class",
                    "interface_declaration" => "interface",
                    "enum_declaration" => "enum",
                    _ => return None,
                };
                
                let mut sig = String::from(keyword);
                sig.push(' ');
                
                if let Some(name) = node.child_by_field_name("name") {
                    if let Ok(text) = name.utf8_text(source) {
                        sig.push_str(text);
                    }
                }
                
                Some(sig)
            }
            _ => None,
        }
    }

    fn extract_docstring(&self, node: Node, source: &[u8]) -> Option<String> {
        let parent = node.parent()?;
        let node_index = (0..parent.named_child_count())
            .find(|&i| parent.named_child(i).map(|c| c.id()) == Some(node.id()))?;

        if node_index > 0 {
            if let Some(prev) = parent.named_child(node_index - 1) {
                if prev.kind() == "block_comment" {
                    if let Ok(text) = prev.utf8_text(source) {
                        if text.starts_with("/**") {
                            return Some(text.to_string());
                        }
                    }
                }
            }
        }
        None
    }

    fn classify(&self, node: Node) -> ChunkKind {
        match node.kind() {
            "method_declaration" | "constructor_declaration" => ChunkKind::Method,
            "class_declaration" | "record_declaration" => ChunkKind::Class,
            "interface_declaration" => ChunkKind::Interface,
            "enum_declaration" => ChunkKind::Enum,
            "annotation_type_declaration" => ChunkKind::Interface,
            _ => ChunkKind::Other,
        }
    }
}

/// C/C++ language extractor
pub struct CppExtractor;

impl LanguageExtractor for CppExtractor {
    fn definition_types(&self) -> &[&'static str] {
        &[
            "function_definition",
            "declaration",
            "struct_specifier",
            "class_specifier",
            "enum_specifier",
            "namespace_definition",
            "template_declaration",
        ]
    }

    fn extract_name(&self, node: Node, source: &[u8]) -> Option<String> {
        // C/C++ has complex declarators
        node.child_by_field_name("declarator")
            .and_then(|d| {
                // Navigate through possible pointer/reference declarators
                let mut current = d;
                while let Some(inner) = current.child_by_field_name("declarator") {
                    current = inner;
                }
                // Get the identifier
                if current.kind() == "identifier" || current.kind() == "field_identifier" {
                    current.utf8_text(source).ok().map(String::from)
                } else {
                    // Try name field first
                    if let Some(name_node) = current.child_by_field_name("name") {
                        return name_node.utf8_text(source).ok().map(String::from);
                    }
                    // Find first identifier child manually
                    for i in 0..current.named_child_count() {
                        if let Some(child) = current.named_child(i) {
                            if child.kind() == "identifier" || child.kind() == "field_identifier" {
                                return child.utf8_text(source).ok().map(String::from);
                            }
                        }
                    }
                    None
                }
            })
            .or_else(|| {
                node.child_by_field_name("name")?
                    .utf8_text(source)
                    .ok()
                    .map(String::from)
            })
    }

    fn extract_signature(&self, node: Node, source: &[u8]) -> Option<String> {
        match node.kind() {
            "function_definition" => {
                // Try to get just the declaration part without body
                let mut sig = String::new();
                
                if let Some(ret) = node.child_by_field_name("type") {
                    if let Ok(text) = ret.utf8_text(source) {
                        sig.push_str(text);
                        sig.push(' ');
                    }
                }
                
                if let Some(declarator) = node.child_by_field_name("declarator") {
                    if let Ok(text) = declarator.utf8_text(source) {
                        sig.push_str(text);
                    }
                }
                
                Some(sig)
            }
            "struct_specifier" | "class_specifier" => {
                let keyword = if node.kind() == "struct_specifier" { "struct" } else { "class" };
                let mut sig = String::from(keyword);
                sig.push(' ');
                
                if let Some(name) = node.child_by_field_name("name") {
                    if let Ok(text) = name.utf8_text(source) {
                        sig.push_str(text);
                    }
                }
                
                Some(sig)
            }
            _ => None,
        }
    }

    fn extract_docstring(&self, node: Node, source: &[u8]) -> Option<String> {
        let parent = node.parent()?;
        let node_index = (0..parent.named_child_count())
            .find(|&i| parent.named_child(i).map(|c| c.id()) == Some(node.id()))?;

        if node_index > 0 {
            if let Some(prev) = parent.named_child(node_index - 1) {
                if prev.kind() == "comment" {
                    if let Ok(text) = prev.utf8_text(source) {
                        if text.starts_with("/**") || text.starts_with("///") {
                            return Some(text.to_string());
                        }
                    }
                }
            }
        }
        None
    }

    fn classify(&self, node: Node) -> ChunkKind {
        match node.kind() {
            "function_definition" => ChunkKind::Function,
            "struct_specifier" => ChunkKind::Struct,
            "class_specifier" => ChunkKind::Class,
            "enum_specifier" => ChunkKind::Enum,
            "namespace_definition" => ChunkKind::Mod,
            _ => ChunkKind::Other,
        }
    }
}

/// Ruby language extractor
pub struct RubyExtractor;

impl LanguageExtractor for RubyExtractor {
    fn definition_types(&self) -> &[&'static str] {
        &[
            "method",
            "singleton_method",
            "class",
            "module",
        ]
    }

    fn extract_name(&self, node: Node, source: &[u8]) -> Option<String> {
        node.child_by_field_name("name")?
            .utf8_text(source)
            .ok()
            .map(String::from)
    }

    fn extract_signature(&self, node: Node, source: &[u8]) -> Option<String> {
        match node.kind() {
            "method" | "singleton_method" => {
                let mut sig = String::from("def ");
                
                if let Some(name) = node.child_by_field_name("name") {
                    if let Ok(text) = name.utf8_text(source) {
                        sig.push_str(text);
                    }
                }
                
                if let Some(params) = node.child_by_field_name("parameters") {
                    if let Ok(text) = params.utf8_text(source) {
                        sig.push_str(text);
                    }
                }
                
                Some(sig)
            }
            "class" => {
                let mut sig = String::from("class ");
                
                if let Some(name) = node.child_by_field_name("name") {
                    if let Ok(text) = name.utf8_text(source) {
                        sig.push_str(text);
                    }
                }
                
                Some(sig)
            }
            "module" => {
                let mut sig = String::from("module ");
                
                if let Some(name) = node.child_by_field_name("name") {
                    if let Ok(text) = name.utf8_text(source) {
                        sig.push_str(text);
                    }
                }
                
                Some(sig)
            }
            _ => None,
        }
    }

    fn extract_docstring(&self, node: Node, source: &[u8]) -> Option<String> {
        // Ruby uses comments before definitions
        let parent = node.parent()?;
        let node_index = (0..parent.named_child_count())
            .find(|&i| parent.named_child(i).map(|c| c.id()) == Some(node.id()))?;

        if node_index > 0 {
            if let Some(prev) = parent.named_child(node_index - 1) {
                if prev.kind() == "comment" {
                    return prev.utf8_text(source).ok().map(String::from);
                }
            }
        }
        None
    }

    fn classify(&self, node: Node) -> ChunkKind {
        match node.kind() {
            "method" | "singleton_method" => {
                if let Some(parent) = node.parent() {
                    if parent.kind() == "class" || parent.kind() == "module" {
                        return ChunkKind::Method;
                    }
                }
                ChunkKind::Function
            }
            "class" => ChunkKind::Class,
            "module" => ChunkKind::Mod,
            _ => ChunkKind::Other,
        }
    }
}

/// PHP language extractor
pub struct PhpExtractor;

impl LanguageExtractor for PhpExtractor {
    fn definition_types(&self) -> &[&'static str] {
        &[
            "function_definition",
            "method_declaration",
            "class_declaration",
            "interface_declaration",
            "trait_declaration",
            "enum_declaration",
        ]
    }

    fn extract_name(&self, node: Node, source: &[u8]) -> Option<String> {
        node.child_by_field_name("name")?
            .utf8_text(source)
            .ok()
            .map(String::from)
    }

    fn extract_signature(&self, node: Node, source: &[u8]) -> Option<String> {
        match node.kind() {
            "function_definition" | "method_declaration" => {
                let mut sig = String::from("function ");
                
                if let Some(name) = node.child_by_field_name("name") {
                    if let Ok(text) = name.utf8_text(source) {
                        sig.push_str(text);
                    }
                }
                
                if let Some(params) = node.child_by_field_name("parameters") {
                    if let Ok(text) = params.utf8_text(source) {
                        sig.push_str(text);
                    }
                }
                
                Some(sig)
            }
            "class_declaration" => {
                let mut sig = String::from("class ");
                
                if let Some(name) = node.child_by_field_name("name") {
                    if let Ok(text) = name.utf8_text(source) {
                        sig.push_str(text);
                    }
                }
                
                Some(sig)
            }
            "interface_declaration" => {
                let mut sig = String::from("interface ");
                
                if let Some(name) = node.child_by_field_name("name") {
                    if let Ok(text) = name.utf8_text(source) {
                        sig.push_str(text);
                    }
                }
                
                Some(sig)
            }
            "trait_declaration" => {
                let mut sig = String::from("trait ");
                
                if let Some(name) = node.child_by_field_name("name") {
                    if let Ok(text) = name.utf8_text(source) {
                        sig.push_str(text);
                    }
                }
                
                Some(sig)
            }
            _ => None,
        }
    }

    fn extract_docstring(&self, node: Node, source: &[u8]) -> Option<String> {
        let parent = node.parent()?;
        let node_index = (0..parent.named_child_count())
            .find(|&i| parent.named_child(i).map(|c| c.id()) == Some(node.id()))?;

        if node_index > 0 {
            if let Some(prev) = parent.named_child(node_index - 1) {
                if prev.kind() == "comment" {
                    if let Ok(text) = prev.utf8_text(source) {
                        if text.starts_with("/**") {
                            return Some(text.to_string());
                        }
                    }
                }
            }
        }
        None
    }

    fn classify(&self, node: Node) -> ChunkKind {
        match node.kind() {
            "function_definition" => ChunkKind::Function,
            "method_declaration" => ChunkKind::Method,
            "class_declaration" => ChunkKind::Class,
            "interface_declaration" => ChunkKind::Interface,
            "trait_declaration" => ChunkKind::Trait,
            "enum_declaration" => ChunkKind::Enum,
            _ => ChunkKind::Other,
        }
    }
}

/// Bash/Shell language extractor
pub struct BashExtractor;

impl LanguageExtractor for BashExtractor {
    fn definition_types(&self) -> &[&'static str] {
        &[
            "function_definition",
        ]
    }

    fn extract_name(&self, node: Node, source: &[u8]) -> Option<String> {
        node.child_by_field_name("name")?
            .utf8_text(source)
            .ok()
            .map(String::from)
    }

    fn extract_signature(&self, node: Node, source: &[u8]) -> Option<String> {
        if node.kind() == "function_definition" {
            let mut sig = String::from("function ");
            
            if let Some(name) = node.child_by_field_name("name") {
                if let Ok(text) = name.utf8_text(source) {
                    sig.push_str(text);
                }
            }
            
            Some(sig)
        } else {
            None
        }
    }

    fn extract_docstring(&self, node: Node, source: &[u8]) -> Option<String> {
        let parent = node.parent()?;
        let node_index = (0..parent.named_child_count())
            .find(|&i| parent.named_child(i).map(|c| c.id()) == Some(node.id()))?;

        if node_index > 0 {
            if let Some(prev) = parent.named_child(node_index - 1) {
                if prev.kind() == "comment" {
                    return prev.utf8_text(source).ok().map(String::from);
                }
            }
        }
        None
    }

    fn classify(&self, node: Node) -> ChunkKind {
        match node.kind() {
            "function_definition" => ChunkKind::Function,
            _ => ChunkKind::Other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_extractor() {
        assert!(get_extractor(Language::Rust).is_some());
        assert!(get_extractor(Language::Python).is_some());
        assert!(get_extractor(Language::JavaScript).is_some());
        assert!(get_extractor(Language::TypeScript).is_some());
        assert!(get_extractor(Language::CSharp).is_some());
        assert!(get_extractor(Language::Go).is_some());
        assert!(get_extractor(Language::Java).is_some());
        assert!(get_extractor(Language::C).is_some());
        assert!(get_extractor(Language::Cpp).is_some());
        assert!(get_extractor(Language::Ruby).is_some());
        assert!(get_extractor(Language::Php).is_some());
        assert!(get_extractor(Language::Shell).is_some());
        assert!(get_extractor(Language::Markdown).is_none());
    }

    #[test]
    fn test_rust_definition_types() {
        let extractor = RustExtractor;
        let types = extractor.definition_types();

        assert!(types.contains(&"function_item"));
        assert!(types.contains(&"struct_item"));
        assert!(types.contains(&"enum_item"));
        assert!(types.contains(&"impl_item"));
    }

    #[test]
    fn test_python_definition_types() {
        let extractor = PythonExtractor;
        let types = extractor.definition_types();

        assert!(types.contains(&"function_definition"));
        assert!(types.contains(&"class_definition"));
    }

    #[test]
    fn test_go_definition_types() {
        let extractor = GoExtractor;
        let types = extractor.definition_types();

        assert!(types.contains(&"function_declaration"));
        assert!(types.contains(&"method_declaration"));
    }

    #[test]
    fn test_java_definition_types() {
        let extractor = JavaExtractor;
        let types = extractor.definition_types();

        assert!(types.contains(&"method_declaration"));
        assert!(types.contains(&"class_declaration"));
    }

    #[test]
    fn test_csharp_definition_types() {
        let extractor = CSharpExtractor;
        let types = extractor.definition_types();

        assert!(types.contains(&"method_declaration"));
        assert!(types.contains(&"class_declaration"));
        assert!(types.contains(&"interface_declaration"));
        assert!(types.contains(&"namespace_declaration"));
    }

    #[test]
    fn test_cpp_definition_types() {
        let extractor = CppExtractor;
        let types = extractor.definition_types();

        assert!(types.contains(&"function_definition"));
        assert!(types.contains(&"class_specifier"));
        assert!(types.contains(&"struct_specifier"));
    }

    #[test]
    fn test_ruby_definition_types() {
        let extractor = RubyExtractor;
        let types = extractor.definition_types();

        assert!(types.contains(&"method"));
        assert!(types.contains(&"class"));
        assert!(types.contains(&"module"));
    }

    #[test]
    fn test_php_definition_types() {
        let extractor = PhpExtractor;
        let types = extractor.definition_types();

        assert!(types.contains(&"function_definition"));
        assert!(types.contains(&"class_declaration"));
        assert!(types.contains(&"trait_declaration"));
    }

    #[test]
    fn test_bash_definition_types() {
        let extractor = BashExtractor;
        let types = extractor.definition_types();

        assert!(types.contains(&"function_definition"));
    }
}
