//! JSON Schema sanitisation for tool declarations.
//!
//! The upstream accepts a narrow subset of JSON Schema and rejects requests
//! carrying anything else with a generic 400. Client-supplied tool schemas come
//! from arbitrary frameworks and routinely contain `$ref`, `anyOf`, `oneOf`,
//! `const`, `additionalProperties`, and constraint keywords that all have to go.
//!
//! The reference implementation's approach is the right one and is reproduced
//! here: rather than dropping information, **fold it into the `description`**.
//! A `pattern`, a `format`, or the alternatives of a `oneOf` are all things the
//! model benefits from knowing, and they are perfectly expressible in prose.
//! Silently deleting them produces a tool the model calls incorrectly.
//!
//! Supported output keywords, and nothing else:
//! `type`, `description`, `properties`, `required`, `items`, `enum`, `nullable`.

use serde_json::{Map, Value, json};

/// Keywords kept as-is. Everything else is dropped or folded into `description`.
const ALLOWED: &[&str] = &[
    "type",
    "description",
    "properties",
    "required",
    "items",
    "enum",
    "nullable",
];

/// Constraint keywords moved into the description rather than discarded.
const CONSTRAINT_HINTS: &[&str] = &[
    "format",
    "pattern",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minProperties",
    "maxProperties",
    "default",
    "examples",
];

/// Placeholder injected into an object schema that declares no properties.
///
/// The upstream rejects an object schema with an empty `properties` map, but a
/// parameterless tool is legitimate. A single optional reason field satisfies the
/// validator without changing how the model calls the tool.
const EMPTY_SCHEMA_PLACEHOLDER: &str = "reason";
const EMPTY_SCHEMA_PLACEHOLDER_DESCRIPTION: &str = "Reason for calling this tool";

/// Sanitise a tool parameter schema.
///
/// Returns `None` when the input is not usable as a schema at all, which the
/// caller turns into a parameterless declaration.
pub fn sanitize(schema: &Value) -> Option<Value> {
    match schema {
        Value::Object(_) => Some(transform(schema, 0)),
        // A boolean schema (`true`/`false`) has no Gemini equivalent.
        _ => None,
    }
}

/// Maximum recursion depth, guarding against a cyclic or pathologically nested
/// client schema.
const MAX_DEPTH: usize = 32;

fn transform(schema: &Value, depth: usize) -> Value {
    if depth > MAX_DEPTH {
        return json!({ "type": "STRING" });
    }

    let Some(object) = schema.as_object() else {
        return json!({ "type": "STRING" });
    };

    let mut hints: Vec<String> = Vec::new();
    let mut merged = object.clone();

    // 1. Resolve $ref into a hint. The referenced definition is not available
    //    here, so pointing the model at it is the most that can be done.
    if let Some(reference) = object.get("$ref").and_then(Value::as_str) {
        let name = reference.rsplit('/').next().unwrap_or(reference);
        hints.push(format!("See: {name}"));
        merged.remove("$ref");
    }

    // 2. Merge allOf members, since the upstream has no intersection type.
    //
    //    `properties` accumulates across members — a first-wins rule here would
    //    silently drop every property after the first member, which is the
    //    normal case for a composed schema. Other keys are first-wins, so a more
    //    specific outer constraint is not overwritten by an inner default.
    if let Some(members) = object.get("allOf").and_then(Value::as_array).cloned() {
        let mut accumulated = Map::new();
        for member in &members {
            let Some(member_object) = member.as_object() else {
                continue;
            };
            for (key, value) in member_object {
                if key == "properties" {
                    if let Some(properties) = value.as_object() {
                        for (name, schema) in properties {
                            accumulated.insert(name.clone(), schema.clone());
                        }
                    }
                } else {
                    merged.entry(key.clone()).or_insert_with(|| value.clone());
                }
            }
        }
        if !accumulated.is_empty() {
            // An outer property declaration outranks an inherited one.
            if let Some(outer) = merged.get("properties").and_then(Value::as_object) {
                for (name, schema) in outer {
                    accumulated.insert(name.clone(), schema.clone());
                }
            }
            merged.insert("properties".into(), Value::Object(accumulated));
        }
        merged.remove("allOf");
    }

    // 3. Collapse anyOf/oneOf to the single best option. The upstream has no
    //    union type; picking the most useful branch and describing the rest
    //    keeps the model informed without an invalid schema.
    if let Some(alternatives) = object
        .get("anyOf")
        .or_else(|| object.get("oneOf"))
        .and_then(Value::as_array)
        .cloned()
    {
        if let Some(best) = alternatives.iter().max_by_key(|option| score_option(option)) {
            for (key, value) in best.as_object().into_iter().flatten() {
                merged.entry(key.clone()).or_insert_with(|| value.clone());
            }
        }
        let names: Vec<String> = alternatives
            .iter()
            .map(describe_option)
            .filter(|name| !name.is_empty())
            .collect();
        if !names.is_empty() {
            hints.push(format!("Accepts: {}", names.join(" | ")));
        }
        merged.remove("anyOf");
        merged.remove("oneOf");
    }

    // 4. Fold `const` into an enum, which the upstream does understand.
    if let Some(constant) = merged.remove("const") {
        let already_enum = merged.contains_key("enum");
        if !already_enum {
            merged.insert("enum".into(), Value::Array(vec![constant]));
        }
    }

    // 5. Collapse `type: ["string", "null"]` to the non-null member, noting the
    //    optionality separately.
    let mut nullable = object.get("nullable").and_then(Value::as_bool).unwrap_or(false);
    if let Some(Value::Array(types)) = merged.get("type").cloned() {
        let non_null: Vec<&Value> = types
            .iter()
            .filter(|entry| entry.as_str() != Some("null"))
            .collect();
        nullable |= types.iter().any(|entry| entry.as_str() == Some("null"));
        match non_null.as_slice() {
            [single] => {
                merged.insert("type".into(), (*single).clone());
            }
            [] => {
                merged.insert("type".into(), Value::String("string".into()));
            }
            _ => {
                // Several concrete types remain. Prefer an object or array,
                // which carry more structure than a scalar.
                let chosen = non_null
                    .iter()
                    .max_by_key(|entry| score_type_name(entry.as_str().unwrap_or("")))
                    .expect("non-empty");
                hints.push(format!(
                    "One of: {}",
                    non_null
                        .iter()
                        .filter_map(|entry| entry.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
                merged.insert("type".into(), (*chosen).clone());
            }
        }
    }

    // 6. Move constraint keywords into the description.
    for keyword in CONSTRAINT_HINTS {
        if let Some(value) = merged.remove(*keyword)
            && let Some(rendered) = render_constraint(keyword, &value)
        {
            hints.push(rendered);
        }
    }

    // 7. Recurse into children before pruning, so nested schemas are normalised.
    if let Some(Value::Object(properties)) = merged.get("properties").cloned() {
        let normalised: Map<String, Value> = properties
            .iter()
            .map(|(name, schema)| (name.clone(), transform(schema, depth + 1)))
            .collect();
        merged.insert("properties".into(), Value::Object(normalised));
    }

    if let Some(items) = merged.get("items").cloned() {
        merged.insert("items".into(), transform_items(&items, depth));
    }

    // 8. Drop everything not on the allowlist.
    let keys: Vec<String> = merged.keys().cloned().collect();
    for key in keys {
        if !ALLOWED.contains(&key.as_str()) {
            merged.remove(&key);
        }
    }

    // 9. Normalise the type name and enforce the invariants that go with it.
    let type_name = merged
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("string")
        .to_ascii_uppercase();
    let type_name = match type_name.as_str() {
        "NULL" => "STRING".to_string(),
        other => other.to_string(),
    };
    merged.insert("type".into(), Value::String(type_name.clone()));

    match type_name.as_str() {
        "OBJECT" => {
            ensure_object_properties(&mut merged);
        }
        "ARRAY" => {
            let items = merged
                .remove("items")
                .unwrap_or_else(|| json!({ "type": "STRING" }));
            merged.insert("items".into(), items);
            // Arrays carry no properties or required list.
            merged.remove("properties");
            merged.remove("required");
        }
        _ => {
            merged.remove("properties");
            merged.remove("required");
            merged.remove("items");
        }
    }

    // 10. A required list naming properties that do not exist is a hard error
    //     upstream, so filter it against the final property set.
    if let Some(Value::Array(required)) = merged.get("required").cloned() {
        let present: Vec<Value> = match merged.get("properties").and_then(Value::as_object) {
            Some(properties) => required
                .into_iter()
                .filter(|name| {
                    name.as_str()
                        .is_some_and(|name| properties.contains_key(name))
                })
                .collect(),
            None => Vec::new(),
        };
        if present.is_empty() {
            merged.remove("required");
        } else {
            merged.insert("required".into(), Value::Array(present));
        }
    }

    if nullable {
        merged.insert("nullable".into(), Value::Bool(true));
    }

    // 11. Fold accumulated hints into the description.
    if !hints.is_empty() {
        let existing = merged
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let combined = if existing.is_empty() {
            hints.join(". ")
        } else {
            format!("{existing} ({})", hints.join(". "))
        };
        merged.insert("description".into(), Value::String(combined));
    }

    Value::Object(merged)
}

/// Normalise an `items` schema, including the tuple form.
fn transform_items(items: &Value, depth: usize) -> Value {
    match items {
        // Tuple form: pick the most structured entry. The upstream has no
        // positional tuple type.
        Value::Array(entries) => entries
            .iter()
            .max_by_key(|entry| score_option(entry))
            .map(|best| transform(best, depth + 1))
            .unwrap_or_else(|| json!({ "type": "STRING" })),
        other => transform(other, depth + 1),
    }
}

/// Guarantee an object schema has a non-empty `properties` map.
fn ensure_object_properties(schema: &mut Map<String, Value>) {
    let is_empty = schema
        .get("properties")
        .and_then(Value::as_object)
        .map(Map::is_empty)
        .unwrap_or(true);

    if is_empty {
        let mut placeholder = Map::new();
        placeholder.insert(
            "type".into(),
            Value::String("STRING".into()),
        );
        placeholder.insert(
            "description".into(),
            Value::String(EMPTY_SCHEMA_PLACEHOLDER_DESCRIPTION.into()),
        );
        let mut properties = Map::new();
        properties.insert(EMPTY_SCHEMA_PLACEHOLDER.into(), Value::Object(placeholder));
        schema.insert("properties".into(), Value::Object(properties));
    }
}

/// Rank an alternative so the most structured one can be chosen.
///
/// An object with properties beats an array, which beats any scalar, which beats
/// `null`. Matching the reference's scoring keeps behaviour consistent with the
/// implementations operators are migrating from.
fn score_option(option: &Value) -> i32 {
    let Some(object) = option.as_object() else {
        return 0;
    };
    if object.get("type").and_then(Value::as_str) == Some("null") {
        return 0;
    }
    if object
        .get("properties")
        .and_then(Value::as_object)
        .is_some_and(|properties| !properties.is_empty())
    {
        return 3;
    }
    match object.get("type").and_then(Value::as_str) {
        Some("object") => 3,
        Some("array") => 2,
        Some(_) => 1,
        None => 1,
    }
}

fn score_type_name(name: &str) -> i32 {
    match name {
        "object" => 3,
        "array" => 2,
        "null" => 0,
        _ => 1,
    }
}

/// Render an alternative as a short label for the `Accepts:` hint.
fn describe_option(option: &Value) -> String {
    let Some(object) = option.as_object() else {
        return String::new();
    };
    match object.get("type").and_then(Value::as_str) {
        Some("null") => "null".into(),
        Some(kind) => match object.get("enum").and_then(Value::as_array) {
            Some(values) => {
                let rendered: Vec<String> = values
                    .iter()
                    .map(|value| match value {
                        Value::String(text) => text.clone(),
                        other => other.to_string(),
                    })
                    .collect();
                format!("{kind}({})", rendered.join(","))
            }
            None => kind.to_string(),
        },
        None => String::new(),
    }
}

/// Render a constraint keyword as prose.
fn render_constraint(keyword: &str, value: &Value) -> Option<String> {
    let rendered = match value {
        Value::String(text) => text.clone(),
        Value::Array(items) => items
            .iter()
            .map(|item| match item {
                Value::String(text) => text.clone(),
                other => other.to_string(),
            })
            .collect::<Vec<_>>()
            .join(", "),
        other => other.to_string(),
    };

    if rendered.is_empty() {
        return None;
    }

    let label = match keyword {
        "format" => "Format",
        "pattern" => "Pattern",
        "minLength" => "Min length",
        "maxLength" => "Max length",
        "minimum" => "Minimum",
        "maximum" => "Maximum",
        "exclusiveMinimum" => "Exclusive minimum",
        "exclusiveMaximum" => "Exclusive maximum",
        "multipleOf" => "Multiple of",
        "minItems" => "Min items",
        "maxItems" => "Max items",
        "uniqueItems" => "Items must be unique",
        "minProperties" => "Min properties",
        "maxProperties" => "Max properties",
        "default" => "Default",
        "examples" => "Examples",
        _ => return None,
    };

    if keyword == "uniqueItems" {
        return Some(label.to_string());
    }
    Some(format!("{label}: {rendered}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sanitize_json(schema: Value) -> Value {
        sanitize(&schema).expect("object schemas are always usable")
    }

    #[test]
    fn simple_object_shape_is_preserved() {
        let result = sanitize_json(json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name" }
            },
            "required": ["city"]
        }));

        assert_eq!(result["type"], "OBJECT");
        assert_eq!(result["properties"]["city"]["type"], "STRING");
        assert_eq!(result["properties"]["city"]["description"], "City name");
        assert_eq!(result["required"], json!(["city"]));
    }

    #[test]
    fn types_are_uppercased() {
        let result = sanitize_json(json!({
            "type": "object",
            "properties": {
                "n": { "type": "number" },
                "b": { "type": "boolean" },
                "i": { "type": "integer" }
            }
        }));
        assert_eq!(result["properties"]["n"]["type"], "NUMBER");
        assert_eq!(result["properties"]["b"]["type"], "BOOLEAN");
        assert_eq!(result["properties"]["i"]["type"], "INTEGER");
    }

    #[test]
    fn ref_is_replaced_with_a_pointer_hint() {
        let result = sanitize_json(json!({
            "type": "object",
            "properties": {
                "address": { "$ref": "#/$defs/Address" }
            }
        }));
        let address = &result["properties"]["address"];
        assert!(address.get("$ref").is_none());
        let description = address["description"].as_str().unwrap();
        assert!(description.contains("Address"), "got: {description}");
    }

    #[test]
    fn all_of_members_are_merged() {
        let result = sanitize_json(json!({
            "type": "object",
            "allOf": [
                { "properties": { "a": { "type": "string" } } },
                { "properties": { "b": { "type": "number" } } }
            ]
        }));
        assert!(result.get("allOf").is_none());
        assert_eq!(result["properties"]["a"]["type"], "STRING");
        assert_eq!(result["properties"]["b"]["type"], "NUMBER");
    }

    #[test]
    fn all_of_does_not_overwrite_outer_keys() {
        let result = sanitize_json(json!({
            "type": "string",
            "description": "outer",
            "allOf": [{ "description": "inner" }]
        }));
        assert_eq!(result["description"], "outer");
    }

    #[test]
    fn one_of_collapses_to_the_most_structured_option() {
        let result = sanitize_json(json!({
            "type": "object",
            "properties": {
                "value": {
                    "oneOf": [
                        { "type": "null" },
                        { "type": "string" },
                        { "type": "object", "properties": { "x": { "type": "string" } } }
                    ]
                }
            }
        }));
        let value = &result["properties"]["value"];
        assert!(value.get("oneOf").is_none());
        assert_eq!(value["type"], "OBJECT", "an object option must win");
    }

    #[test]
    fn any_of_records_the_alternatives_in_the_description() {
        let result = sanitize_json(json!({
            "type": "object",
            "properties": {
                "value": { "anyOf": [{ "type": "string" }, { "type": "number" }] }
            }
        }));
        let description = result["properties"]["value"]["description"]
            .as_str()
            .unwrap();
        assert!(description.contains("Accepts:"), "got: {description}");
        assert!(description.contains("string"));
        assert!(description.contains("number"));
    }

    #[test]
    fn const_becomes_a_single_value_enum() {
        let result = sanitize_json(json!({
            "type": "object",
            "properties": { "kind": { "const": "weather" } }
        }));
        let kind = &result["properties"]["kind"];
        assert!(kind.get("const").is_none());
        assert_eq!(kind["enum"], json!(["weather"]));
    }

    #[test]
    fn const_does_not_overwrite_an_existing_enum() {
        let result = sanitize_json(json!({
            "type": "object",
            "properties": { "kind": { "enum": ["a", "b"], "const": "a" } }
        }));
        assert_eq!(result["properties"]["kind"]["enum"], json!(["a", "b"]));
        assert!(result["properties"]["kind"].get("const").is_none());
    }

    #[test]
    fn nullable_type_arrays_collapse_to_the_non_null_member() {
        let result = sanitize_json(json!({
            "type": "object",
            "properties": { "note": { "type": ["string", "null"] } },
            "required": ["note"]
        }));
        let note = &result["properties"]["note"];
        assert_eq!(note["type"], "STRING");
        assert_eq!(note["nullable"], true);
    }

    #[test]
    fn multi_type_arrays_pick_the_richest_member_and_record_the_rest() {
        let result = sanitize_json(json!({
            "type": "object",
            "properties": { "v": { "type": ["string", "object", "null"] } }
        }));
        let v = &result["properties"]["v"];
        assert_eq!(v["type"], "OBJECT");
        let description = v["description"].as_str().unwrap();
        assert!(description.contains("One of:"), "got: {description}");
    }

    #[test]
    fn constraints_move_into_the_description() {
        let result = sanitize_json(json!({
            "type": "object",
            "properties": {
                "code": { "type": "string", "pattern": "^[A-Z]{3}$", "minLength": 3 }
            }
        }));
        let code = &result["properties"]["code"];
        assert!(code.get("pattern").is_none(), "pattern must not be sent");
        assert!(code.get("minLength").is_none(), "minLength must not be sent");
        let description = code["description"].as_str().unwrap();
        assert!(description.contains("Pattern: ^[A-Z]{3}$"), "got: {description}");
        assert!(description.contains("Min length: 3"), "got: {description}");
    }

    #[test]
    fn unsupported_keywords_are_removed() {
        let result = sanitize_json(json!({
            "type": "object",
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$defs": { "Unused": { "type": "string" } },
            "additionalProperties": false,
            "title": "MyTool",
            "properties": { "a": { "type": "string" } }
        }));

        for key in ["$schema", "$defs", "additionalProperties", "title"] {
            assert!(result.get(key).is_none(), "{key} should have been removed");
        }
        let keys: Vec<&String> = result.as_object().unwrap().keys().collect();
        for key in keys {
            assert!(ALLOWED.contains(&key.as_str()), "unexpected key {key}");
        }
    }

    #[test]
    fn arrays_get_exactly_one_object_items() {
        let result = sanitize_json(json!({
            "type": "object",
            "properties": { "tags": { "type": "array" } }
        }));
        let items = &result["properties"]["tags"]["items"];
        assert!(items.is_object());
        assert_eq!(items["type"], "STRING");
    }

    #[test]
    fn tuple_items_collapse_to_the_most_structured_entry() {
        let result = sanitize_json(json!({
            "type": "object",
            "properties": {
                "pair": {
                    "type": "array",
                    "items": [
                        { "type": "string" },
                        { "type": "object", "properties": { "x": { "type": "string" } } }
                    ]
                }
            }
        }));
        assert_eq!(result["properties"]["pair"]["items"]["type"], "OBJECT");
    }

    #[test]
    fn non_object_types_drop_object_only_keywords() {
        let result = sanitize_json(json!({
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "properties": { "stray": { "type": "string" } },
                    "required": ["stray"]
                }
            }
        }));
        let label = &result["properties"]["label"];
        assert!(label.get("properties").is_none());
        assert!(label.get("required").is_none());
    }

    #[test]
    fn empty_object_schema_gets_a_placeholder_property() {
        let result = sanitize_json(json!({
            "type": "object",
            "properties": {}
        }));
        let properties = result["properties"].as_object().unwrap();
        assert_eq!(properties.len(), 1);
        assert!(properties.contains_key(EMPTY_SCHEMA_PLACEHOLDER));
        assert_eq!(properties[EMPTY_SCHEMA_PLACEHOLDER]["type"], "STRING");
    }

    #[test]
    fn object_without_properties_gets_a_placeholder_property() {
        let result = sanitize_json(json!({ "type": "object" }));
        assert!(result["properties"].is_object());
        assert!(!result["properties"].as_object().unwrap().is_empty());
    }

    #[test]
    fn required_names_that_do_not_exist_are_dropped() {
        // A required entry with no matching property is a hard upstream error.
        let result = sanitize_json(json!({
            "type": "object",
            "properties": { "a": { "type": "string" } },
            "required": ["a", "ghost"]
        }));
        assert_eq!(result["required"], json!(["a"]));
    }

    #[test]
    fn required_becomes_absent_when_nothing_matches() {
        let result = sanitize_json(json!({
            "type": "object",
            "properties": { "a": { "type": "string" } },
            "required": ["ghost"]
        }));
        assert!(result.get("required").is_none());
    }

    #[test]
    fn missing_type_defaults_to_string() {
        let result = sanitize_json(json!({ "description": "anything goes" }));
        assert_eq!(result["type"], "STRING");
    }

    #[test]
    fn non_object_schema_is_rejected() {
        // A boolean schema has no Gemini equivalent; the caller substitutes a
        // parameterless declaration.
        assert!(sanitize(&json!(true)).is_none());
        assert!(sanitize(&json!("string")).is_none());
    }

    #[test]
    fn deeply_nested_schemas_terminate() {
        // Guards the recursion bound against a pathological client schema.
        let mut schema = json!({ "type": "string" });
        for _ in 0..200 {
            schema = json!({
                "type": "object",
                "properties": { "nested": schema }
            });
        }
        let result = sanitize_json(schema);
        assert!(result.is_object());
    }

    #[test]
    fn enum_survives_untouched() {
        let result = sanitize_json(json!({
            "type": "object",
            "properties": { "unit": { "type": "string", "enum": ["c", "f"] } }
        }));
        assert_eq!(result["properties"]["unit"]["enum"], json!(["c", "f"]));
    }

    #[test]
    fn nested_objects_are_sanitised_recursively() {
        let result = sanitize_json(json!({
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {
                        "inner": { "type": "string", "pattern": "^x$" }
                    }
                }
            }
        }));
        let inner = &result["properties"]["outer"]["properties"]["inner"];
        assert!(inner.get("pattern").is_none());
        assert!(
            inner["description"].as_str().unwrap().contains("Pattern"),
            "nested constraints must be folded too"
        );
    }

    #[test]
    fn real_world_anthropic_style_schema_survives() {
        // The shape a typical agent framework sends: $schema, $defs, refs,
        // anyOf nullables, and constraints.
        let result = sanitize_json(json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File to read",
                    "minLength": 1
                },
                "encoding": {
                    "anyOf": [{ "type": "string" }, { "type": "null" }],
                    "default": "utf-8"
                }
            },
            "required": ["path"],
            "additionalProperties": false
        }));

        assert_eq!(result["type"], "OBJECT");
        assert_eq!(result["properties"]["path"]["type"], "STRING");
        assert_eq!(result["required"], json!(["path"]));

        let encoding = &result["properties"]["encoding"];
        assert!(encoding["description"]
            .as_str()
            .unwrap()
            .contains("Default: utf-8"));
        assert!(result.get("additionalProperties").is_none());
    }
}
