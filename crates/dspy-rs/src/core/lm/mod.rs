pub mod chat;
pub mod client_registry;
mod tool_loop;
pub mod usage;

pub use chat::{Chat, ContentBlock, Message, Role};
pub use client_registry::{
    CompletionProvider, CustomCompletionModel, LMClient, TestCompletionModel,
};
pub use tool_loop::{ToolLoopError, ToolLoopMode, ToolSet};
pub use usage::LmUsage;

use anyhow::Result;
use rig::{
    completion::{AssistantContent, CompletionError, CompletionRequest, CompletionResponse},
    message::ToolCall,
    message::ToolChoice,
    tool::ToolDyn,
};

use bon::Builder;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::{sync::Arc, time::Duration};
use tool_loop::{
    ChoiceAction, ToolLoopResult, ToolRound, classify_choice, decode_prompt_tools,
    format_prompt_tools,
};
use tracing::{debug, trace, warn};

#[cfg(feature = "cache")]
use crate::ResponseCache;
use crate::core::history::{CallProgress, HistoryCall};
use crate::trace::SpanEvent;
#[cfg(feature = "cache")]
use crate::utils::cache::CacheEntry;
#[cfg(all(test, feature = "inflight-dedup"))]
use crate::utils::cache::InflightResponse;
#[cfg(feature = "inflight-dedup")]
use crate::utils::cache::SharedResponse;

#[derive(Clone, Debug)]
pub struct LMResponse {
    /// True when this response was served from the response cache.
    pub cache_hit: bool,
    /// Shared history record, enriched with parsed output by the interpreter.
    pub history: Option<crate::HistoryRecord>,
    /// Assistant message chosen by the provider.
    pub output: Message,
    /// Token usage reported by the provider for this call (aggregate).
    pub usage: LmUsage,
    /// Chat history including the freshly appended assistant response.
    pub chat: Chat,
    /// Tool calls made by the provider. Deprecated by `events`.
    pub tool_calls: Vec<ToolCall>,
    /// Tool executions made by the provider. Deprecated by `events`.
    pub tool_executions: Vec<String>,
    /// Ordered per-round-trip record: one `Exchange` per provider call with
    /// that round-trip's own usage, `ToolRun` entries interleaved in execution
    /// order. Cache-served responses synthesize a single `Exchange`.
    pub events: Vec<SpanEvent>,
}

/// The data half of an LM: every generation parameter, no live state.
///
/// This is the serializable artifact — what a program file records, what an
/// optimizer can hash, diff, and mutate. The live half ([`LM`]) is built from
/// it with [`LM::from_config`]. `api_key` is deliberately `#[serde(skip)]`:
/// secrets never serialize, and on load the key is resolved from provider env
/// vars at client initialization. Serialized `base_url` values keep the
/// endpoint origin and the public `/v1` API prefix. Other path segments and
/// URL userinfo, queries, and fragments can carry secrets and are omitted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Builder)]
#[builder(finish_fn(vis = "", name = __internal_build))]
pub struct LMConfig {
    /// Set false for models/endpoints without native function calling. Tool
    /// declarations and results then travel as text through the same tool loop.
    #[serde(
        default = "native_tools_default",
        skip_serializing_if = "native_tools_enabled"
    )]
    #[builder(default = true)]
    pub native_tools: bool,
    #[serde(serialize_with = "serialize_public_base_url")]
    pub base_url: Option<String>,
    #[serde(skip)]
    pub api_key: Option<String>,
    #[builder(default = "openai:gpt-4o-mini".to_string())]
    pub model: String,
    #[builder(default = 0.7)]
    pub temperature: f32,
    #[builder(default = 512)]
    pub max_tokens: u32,
    /// Maximum tool-execution rounds in the automatic LM loop, including the
    /// initial round. Parallel tool calls in one response count as one round.
    /// A final completion is allowed after the last round; further tool requests
    /// fail before execution. `0` disables automatic tool execution.
    /// Caller-managed loops and IR agent limits are controlled separately.
    #[builder(default = 10)]
    pub max_tool_iterations: u32,
    /// Additional attempts after a transient failure (429/5xx/network/timeout).
    /// `0` disables retries entirely.
    #[builder(default = 2)]
    pub max_retries: u32,
    /// Base delay for exponential backoff between retries. Attempt `n` waits
    /// `base * 2^n` plus up to 50% random jitter.
    #[builder(default = 250)]
    pub retry_base_delay_ms: u64,
    #[builder(default = false)]
    pub cache: bool,
}

fn serialize_public_base_url<S>(url: &Option<String>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    url.as_deref()
        .map(public_model_base_url)
        .serialize(serializer)
}

/// Keep the endpoint origin stable for trace identities.
pub(crate) fn url_origin(url: &str) -> String {
    let (scheme, rest) = match url.split_once("://") {
        Some((scheme, rest)) => (scheme, rest),
        None => ("", url),
    };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or(rest);
    let host = authority.rsplit('@').next().unwrap_or(authority);
    if scheme.is_empty() {
        host.to_string()
    } else {
        format!("{scheme}://{host}")
    }
}

/// Reject malformed URLs and retain only a fixed public API prefix in artifacts.
pub(crate) fn public_model_base_url(url: &str) -> String {
    let has_scheme = url.contains("://");
    let parsed = if has_scheme {
        reqwest::Url::parse(url)
    } else {
        reqwest::Url::parse(&format!("http://{url}"))
    };
    let Ok(parsed) = parsed else {
        return "<invalid URL>".to_string();
    };
    if parsed.host_str().is_none() {
        return "<invalid URL>".to_string();
    }
    let mut origin = parsed.origin().ascii_serialization();
    if matches!(parsed.path(), "/v1" | "/v1/") {
        origin.push_str(parsed.path());
    }
    if has_scheme {
        origin
    } else {
        origin
            .strip_prefix("http://")
            .unwrap_or("<invalid URL>")
            .to_string()
    }
}

/// Per-call structured output request. Unsupported clients ignore the schema and
/// callers retain their delimiter prompt/parser path.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LMCallOptions {
    pub output_schema: Option<serde_json::Value>,
    #[serde(default)]
    pub json_mode: bool,
    /// Explicit capability override for custom OpenAI-compatible endpoints.
    #[serde(default)]
    pub force_output_schema: bool,
}

impl LMCallOptions {
    pub fn is_json(&self) -> bool {
        self.output_schema.is_some() || self.json_mode
    }
}

impl Default for LMConfig {
    fn default() -> Self {
        LMConfig::builder().__internal_build()
    }
}

fn native_tools_default() -> bool {
    true
}

fn native_tools_enabled(enabled: &bool) -> bool {
    *enabled
}

/// The live half: an [`LMConfig`] plus the initialized provider client and
/// response cache. Constructed via [`LM::builder()`] or [`LM::from_config`].
#[derive(Clone)]
pub struct LM {
    pub history: crate::HistoryBuffer,
    pub config: LMConfig,
    /// Response cache storage, available only with the `cache` Cargo feature.
    #[cfg(feature = "cache")]
    pub cache_handler: Option<Arc<ResponseCache>>,
    client: Option<Arc<LMClient>>,
    client_id: u64,
    custom_output_schema: bool,
    scalar_output_schema: bool,
}

impl Default for LM {
    fn default() -> Self {
        tokio::runtime::Handle::current().block_on(async { Self::builder().build().await.unwrap() })
    }
}

impl LM {
    fn next_client_id() -> u64 {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        NEXT.fetch_add(1, Ordering::Relaxed)
    }

    pub(crate) fn binding_identity(&self) -> (u64, u64, bool, bool) {
        (
            self.client_id,
            crate::trace::ModelEntry::from_config(&self.config).config_hash,
            self.custom_output_schema,
            self.scalar_output_schema,
        )
    }

    pub fn with_native_tools(mut self, enabled: bool) -> Self {
        self.config.native_tools = enabled;
        self
    }
    /// Enables native output formatting for an explicitly verified custom OpenAI-compatible endpoint.
    pub fn with_custom_output_schema(mut self, enabled: bool) -> Self {
        self.custom_output_schema = enabled;
        self
    }

    /// Opts into root JSON scalars on a verified schema-capable endpoint.
    /// Tool-free signatures with one string output may omit its object wrapper.
    /// Leave disabled for providers whose schema subset requires an object root.
    pub fn with_scalar_output_schema(mut self, enabled: bool) -> Self {
        self.scalar_output_schema = enabled;
        self
    }

    /// Reports an explicitly enabled scalar capability on a schema-capable client.
    pub fn supports_scalar_output_schema(&self) -> bool {
        self.scalar_output_schema && self.supports_output_schema()
    }
    /// Reports schema support based on the initialized provider client.
    pub fn supports_output_schema(&self) -> bool {
        if self.config.base_url.is_some() {
            self.custom_output_schema
                && matches!(
                    self.client.as_deref(),
                    Some(LMClient::OpenAI(_) | LMClient::Test(_))
                )
        } else {
            self.client
                .as_deref()
                .is_some_and(LMClient::supports_output_schema)
        }
    }
    pub fn supports_json_mode(&self) -> bool {
        if self.config.base_url.is_some() {
            self.custom_output_schema
                && matches!(
                    self.client.as_deref(),
                    Some(LMClient::OpenAI(_) | LMClient::Test(_))
                )
        } else {
            self.client
                .as_deref()
                .is_some_and(LMClient::supports_json_mode)
        }
    }
    /// Entry point for fluent construction: `LM::builder().model(...).build().await`.
    /// The builder collects an [`LMConfig`]; `build()` initializes the live client.
    pub fn builder() -> LMConfigBuilder {
        LMConfig::builder()
    }

    /// Builds the live [`LM`] from a config, initializing the provider client and
    /// optional response cache.
    ///
    /// Supports 3 build cases:
    /// 1. OpenAI-compatible with auth: `base_url` + `api_key` provided
    ///    → Uses OpenAI client with custom base URL
    /// 2. Local OpenAI-compatible: `base_url` only (no `api_key`)
    ///    → Uses OpenAI client for vLLM/local servers (dummy key)
    /// 3. Provider via model string: no `base_url`, model in "provider:model" format
    ///    → Uses provider-specific client (openai, anthropic, gemini, etc.)
    #[tracing::instrument(
        name = "dsrs.lm.from_config",
        level = "debug",
        skip(config),
        fields(
            model = %config.model,
            base_url_present = config.base_url.is_some(),
            api_key_present = config.api_key.is_some(),
            cache_enabled = config.cache,
            max_tokens = config.max_tokens,
            temperature = config.temperature,
            max_tool_iterations = config.max_tool_iterations
        )
    )]
    pub async fn from_config(config: LMConfig) -> Result<Self> {
        validate_cache_setting(config.cache)?;
        // Determine which build case based on what's provided
        let client = match (&config.base_url, &config.api_key, &config.model) {
            // Case 1: OpenAI-compatible with authentication (base_url + api_key)
            // For custom OpenAI-compatible APIs that require API keys
            (Some(base_url), Some(api_key), _) => {
                debug!(build_case = 1, "using openai-compatible client with auth");
                Arc::new(LMClient::from_openai_compatible(
                    base_url,
                    api_key,
                    &config.model,
                )?)
            }
            // Case 2: Local OpenAI-compatible server (base_url only, no api_key)
            // For vLLM, text-generation-inference, and other local OpenAI-compatible servers
            (Some(base_url), None, _) => {
                debug!(build_case = 2, "using local openai-compatible client");
                Arc::new(LMClient::from_local(base_url, &config.model)?)
            }
            // Case 3: Provider via model string (no base_url, model in "provider:model" format)
            // Uses provider-specific clients
            (None, api_key, model) if model.contains(':') => {
                debug!(build_case = 3, "using provider:model client");
                Arc::new(LMClient::from_model_string(model, api_key.as_deref())?)
            }
            // Default case: assume OpenAI provider if no colon in model name
            (None, api_key, model) => {
                debug!(build_case = 4, "defaulting model to openai provider");
                let model_str = if model.contains(':') {
                    model.to_string()
                } else {
                    format!("openai:{}", model)
                };
                Arc::new(LMClient::from_model_string(&model_str, api_key.as_deref())?)
            }
        };

        #[cfg(feature = "cache")]
        let cache_handler = if config.cache {
            debug!("initializing response cache");
            Some(Arc::new(ResponseCache::new().await))
        } else {
            None
        };

        debug!("lm client initialized");
        Ok(LM {
            history: crate::HistoryBuffer::default(),
            config,
            #[cfg(feature = "cache")]
            cache_handler,
            client: Some(client),
            client_id: Self::next_client_id(),
            custom_output_schema: false,
            scalar_output_schema: false,
        })
    }

    /// Copies this LM, applying changes over its existing configuration.
    ///
    /// Generation-only changes retain the provider client (including injected
    /// clients). Model, endpoint, or credential changes initialize a new client.
    /// Copies always start with fresh history and a fresh cache when enabled.
    pub async fn copy_with(&self, overrides: impl FnOnce(&mut LMConfig)) -> Result<Self> {
        let mut config = self.config.clone();
        overrides(&mut config);
        validate_cache_setting(config.cache)?;
        if config.model != self.config.model
            || config.base_url != self.config.base_url
            || config.api_key != self.config.api_key
        {
            return Self::from_config(config).await;
        }
        #[cfg(feature = "cache")]
        let cache_handler = if config.cache {
            Some(Arc::new(ResponseCache::new().await))
        } else {
            None
        };
        Ok(Self {
            history: crate::HistoryBuffer::default(),
            config,
            #[cfg(feature = "cache")]
            cache_handler,
            ..self.clone()
        })
    }

    /// Replaces the provider client and starts fresh history and cache state.
    /// Existing clones retain their client, cache, and history.
    pub async fn with_client(self, client: LMClient) -> Result<Self> {
        validate_cache_setting(self.config.cache)?;
        #[cfg(feature = "cache")]
        let cache_handler = if self.config.cache {
            Some(Arc::new(ResponseCache::new().await))
        } else {
            None
        };
        Ok(LM {
            history: crate::HistoryBuffer::default(),
            #[cfg(feature = "cache")]
            cache_handler,
            client: Some(Arc::new(client)),
            client_id: Self::next_client_id(),
            ..self
        })
    }
}

// Implement build() for all builder states since optional fields don't require setting
impl<S: l_m_config_builder::State> LMConfigBuilder<S> {
    /// Finishes the config and initializes the live client — see [`LM::from_config`].
    #[tracing::instrument(name = "dsrs.lm.build", level = "debug", skip(self))]
    pub async fn build(self) -> Result<LM> {
        let config = self.__internal_build();
        debug!(
            model = %config.model,
            base_url_present = config.base_url.is_some(),
            api_key_present = config.api_key.is_some(),
            cache_enabled = config.cache,
            "building lm"
        );
        LM::from_config(config).await
    }
}

#[derive(Clone, Copy, Debug)]
struct PreparedCallOptions<'a> {
    json_mode: bool,
    output_schema: Option<&'a schemars::Schema>,
}

fn validate_cache_setting(enabled: bool) -> Result<()> {
    if enabled && !cfg!(feature = "cache") {
        return Err(crate::LmCallError::FeatureDisabled { feature: "cache" }.into());
    }
    Ok(())
}

/// Scan all content blocks in a response to find actionable items.
/// Anthropic returns `[Reasoning, ToolCall]` or `[Reasoning, Text]`;
/// OpenAI Responses API returns `[Reasoning, FunctionCall]`.
/// Multiple tool calls in one response are supported (parallel tool calling).
/// Keep provider URLs, credentials, and response bodies out of logs and run errors.
pub(crate) fn safe_completion_error_kind(err: &CompletionError) -> &'static str {
    match err {
        CompletionError::HttpError(_) => "HTTP transport error",
        CompletionError::JsonError(_) => "provider JSON error",
        CompletionError::UrlError(_) => "invalid provider URL",
        CompletionError::RequestError(_) => "request construction error",
        CompletionError::ResponseError(_) => "provider response error",
        CompletionError::ProviderError(_) => "provider error",
    }
}

/// Whether a rig completion error is worth retrying.
///
/// Retry transport failures and HTTP 429/5xx. Permanent HTTP errors and
/// request-construction errors fail immediately. Provider errors are string-typed
/// in rig, so transient markers are matched textually only for that variant.
pub(crate) fn is_retryable_completion_error(err: &CompletionError) -> bool {
    match err {
        CompletionError::HttpError(error) => match error {
            rig::http_client::Error::InvalidStatusCode(status)
            | rig::http_client::Error::InvalidStatusCodeWithMessage(status, _) => {
                status.as_u16() == 429 || status.is_server_error()
            }
            rig::http_client::Error::Instance(source) => source
                .downcast_ref::<reqwest::Error>()
                .is_some_and(|error| error.is_connect() || error.is_timeout() || error.is_body()),
            rig::http_client::Error::StreamEnded => true,
            _ => false,
        },
        CompletionError::ProviderError(message) => {
            let message = message.to_ascii_lowercase();
            [
                "429",
                "rate limit",
                "rate_limit",
                "too many requests",
                "overloaded",
                "timeout",
                "timed out",
                "500",
                "502",
                "503",
                "529",
                "server error",
                "internal error",
                "unavailable",
            ]
            .iter()
            .any(|marker| message.contains(marker))
        }
        _ => false,
    }
}

impl LM {
    /// Builds the response-cache key: a streaming hash over the full message
    /// history plus every generation parameter that changes the completion.
    /// Demos and instructions live inside the messages, so they are covered
    /// automatically. Hashing streams through the `Debug` representation — no
    /// intermediate JSON tree or string is materialized. Uses the same stable
    /// hasher as the trace format's `request_hash`.
    #[cfg(feature = "cache")]
    fn cache_key_for(&self, messages: &Chat, options: &LMCallOptions) -> u64 {
        use crate::utils::hash::{HashWriter, StableHasher};
        use std::hash::Hasher;

        let mut hasher = StableHasher::new();
        hasher.write(self.config.model.as_bytes());
        hasher.write(&[u8::from(self.config.native_tools)]);
        if let Some(base_url) = &self.config.base_url {
            hasher.write(base_url.as_bytes());
        }
        hasher.write(&[
            self.supports_output_schema() as u8,
            self.supports_json_mode() as u8,
        ]);
        let sampling = crate::core::sampling::current_sampling_options();
        hasher.write(
            &crate::core::sampling::effective_temperature(self.config.temperature)
                .to_bits()
                .to_le_bytes(),
        );
        if let Some(id) = sampling.id {
            hasher.write(&(id.len() as u64).to_le_bytes());
            hasher.write(id.as_bytes());
        }
        hasher.write(&self.config.max_tokens.to_le_bytes());
        if let Ok(options) = serde_json::to_vec(options) {
            hasher.write(&options);
        }
        use std::fmt::Write as _;
        let _ = write!(HashWriter(&mut hasher), "{messages:?}");
        hasher.finish()
    }

    fn chat_from_rig_history(system_prompt: String, history: Vec<rig::message::Message>) -> Chat {
        let mut messages = Vec::with_capacity(history.len() + 2);
        if !system_prompt.is_empty() {
            messages.push(Message::system(system_prompt));
        }
        messages.extend(history.into_iter().map(Message::from));
        Chat::new(messages)
    }

    /// Builds a rig completion request from borrowed parts.
    ///
    /// Kept as a separate builder (rather than constructing once) so the retry
    /// loop can rebuild the request per attempt — the success-first-try path
    /// then pays exactly one build, with no defensive clone.
    fn build_completion_request(
        &self,
        system_prompt: &str,
        chat_history: &[rig::message::Message],
        tool_definitions: &[rig::completion::ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: PreparedCallOptions<'_>,
    ) -> CompletionRequest {
        use rig::OneOrMany;
        CompletionRequest {
            model: None,
            preamble: Some(system_prompt.to_string()),
            chat_history: if chat_history.len() == 1 {
                OneOrMany::one(chat_history[0].clone())
            } else {
                OneOrMany::many(chat_history.to_vec()).expect("chat_history should not be empty")
            },
            documents: Vec::new(),
            tools: tool_definitions.to_vec(),
            temperature: Some(
                crate::core::sampling::effective_temperature(self.config.temperature) as f64,
            ),
            max_tokens: Some(self.config.max_tokens as u64),
            tool_choice,
            additional_params: if options.json_mode {
                match self.client.as_deref() {
                    Some(LMClient::Gemini(_)) => Some(
                        serde_json::json!({"generationConfig":{"responseMimeType":"application/json"}}),
                    ),
                    _ => Some(serde_json::json!({"response_format":{"type":"json_object"}})),
                }
            } else {
                None
            },
            output_schema: options.output_schema.cloned(),
        }
    }

    /// Calls the provider, retrying transient failures with jittered exponential backoff.
    ///
    /// Takes a request *builder* so each attempt constructs its own request from
    /// borrowed parts — no whole-request clone on the common no-retry path.
    async fn completion_with_retry<F>(
        &self,
        build_request: F,
        progress: &mut CallProgress,
    ) -> Result<CompletionResponse<()>>
    where
        F: Fn() -> CompletionRequest,
    {
        let client = self
            .client
            .as_ref()
            .ok_or(crate::LmCallError::NotInitialized)?;

        let mut attempt = 0u32;
        loop {
            let mut request = build_request();
            let prompt_tools = !self.config.native_tools && !request.tools.is_empty();
            let tool_required = matches!(&request.tool_choice, Some(ToolChoice::Required));
            if !self.config.native_tools {
                format_prompt_tools(&mut request)?;
            }
            match client.completion(request).await {
                Ok(mut response) => {
                    progress.observe_usage(response.usage.into());
                    if prompt_tools {
                        decode_prompt_tools(&mut response, tool_required)?;
                    }
                    debug!(target: "dsrs::content", choice = ?response.choice, "provider response");
                    return Ok(response);
                }
                Err(err)
                    if attempt < self.config.max_retries && is_retryable_completion_error(&err) =>
                {
                    let backoff = self
                        .config
                        .retry_base_delay_ms
                        .saturating_mul(1u64 << attempt.min(16));
                    // Scope the RNG so it drops before the await (thread_rng is !Send).
                    let jitter = {
                        use rand::Rng;
                        rand::thread_rng().gen_range(0..=backoff / 2)
                    };
                    let delay = Duration::from_millis(backoff.saturating_add(jitter));
                    warn!(
                        attempt = attempt + 1,
                        max_retries = self.config.max_retries,
                        delay_ms = delay.as_millis() as u64,
                        error_kind = safe_completion_error_kind(&err),
                        "retrying transient lm completion failure"
                    );
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                }
                Err(err) => {
                    // Host-supplied providers own the diagnostics in their error sources.
                    let redact_details =
                        !matches!(client.as_ref(), LMClient::Custom(_) | LMClient::Test(_));
                    return Err(crate::LmCallError::from_provider_error(err, redact_details).into());
                }
            }
        }
    }

    pub async fn call(&self, messages: Chat, tools: Vec<Arc<dyn ToolDyn>>) -> Result<LMResponse> {
        self.call_with_options(
            messages,
            tools,
            ToolLoopMode::Auto,
            LMCallOptions::default(),
        )
        .await
    }

    /// Calls the model with additive per-request options; legacy calls use defaults.
    pub async fn call_with_options(
        &self,
        messages: Chat,
        tools: Vec<Arc<dyn ToolDyn>>,
        mode: ToolLoopMode,
        options: LMCallOptions,
    ) -> Result<LMResponse> {
        let toolset = if tools.is_empty() {
            ToolSet::default()
        } else {
            ToolSet::build(&tools).await
        };
        self.call_with_toolset_and_options(messages, &toolset, mode, options)
            .await
    }

    /// [`call`](LM::call) with an explicit tool-loop mode. Builds an ad-hoc
    /// [`ToolSet`] per call — reuse [`call_with_toolset`](LM::call_with_toolset)
    /// with a cached set when calling repeatedly with the same tools.
    pub async fn call_with_tool_loop_mode(
        &self,
        messages: Chat,
        tools: Vec<Arc<dyn ToolDyn>>,
        tool_loop_mode: ToolLoopMode,
    ) -> Result<LMResponse> {
        let toolset = if tools.is_empty() {
            ToolSet::default()
        } else {
            ToolSet::build(&tools).await
        };
        self.call_with_toolset(messages, &toolset, tool_loop_mode)
            .await
    }

    #[tracing::instrument(
        name = "dsrs.lm.call_with_toolset",
        level = "debug",
        skip(self, messages, tools),
        fields(
            model = %self.config.model,
            message_count = messages.len(),
            tool_count = tools.definitions().len(),
            cache_enabled = self.config.cache,
            tool_loop_mode = ?tool_loop_mode
        )
    )]
    pub async fn call_with_toolset(
        &self,
        messages: Chat,
        tools: &ToolSet,
        tool_loop_mode: ToolLoopMode,
    ) -> Result<LMResponse> {
        self.call_with_toolset_and_options(
            messages,
            tools,
            tool_loop_mode,
            LMCallOptions::default(),
        )
        .await
    }

    pub async fn call_with_toolset_and_options(
        &self,
        messages: Chat,
        tools: &ToolSet,
        tool_loop_mode: ToolLoopMode,
        options: LMCallOptions,
    ) -> Result<LMResponse> {
        self.call_with_toolset_and_options_ref(messages, tools, tool_loop_mode, &options)
            .await
    }

    pub(crate) async fn call_with_toolset_and_options_ref(
        &self,
        messages: Chat,
        tools: &ToolSet,
        tool_loop_mode: ToolLoopMode,
        options: &LMCallOptions,
    ) -> Result<LMResponse> {
        let mut history = HistoryCall::begin(&self.config, &messages, &self.history);
        debug!(target: "dsrs::content", prompt = %messages.to_json(), "formatted LM prompt");
        let mut result = self
            .call_with_toolset_inner(
                messages,
                tools,
                tool_loop_mode,
                options,
                &mut history.progress,
            )
            .await;
        history.finish(&mut result).await;
        if let Ok(response) = &result {
            debug!(target: "dsrs::content", response = %response.output.to_json(), "raw LM response");
        }
        result
    }

    async fn call_with_toolset_inner(
        &self,
        messages: Chat,
        tools: &ToolSet,
        tool_loop_mode: ToolLoopMode,
        options: &LMCallOptions,
        progress: &mut CallProgress,
    ) -> Result<LMResponse> {
        validate_cache_setting(self.config.cache)?;
        let output_schema = if self.supports_output_schema() || options.force_output_schema {
            options
                .output_schema
                .as_ref()
                .map(<&schemars::Schema>::try_from)
                .transpose()
                .map_err(|source| crate::LmCallError::InvalidOutputSchema { source })?
        } else {
            None
        };
        if options.json_mode && !(self.supports_json_mode() || options.force_output_schema) {
            return Err(crate::LmCallError::UnsupportedOption {
                option: "JSON mode",
            }
            .into());
        }
        // Response cache: only tool-free calls are cached — tool loops execute
        // side-effectful user code and must not be replayed from cache.
        #[cfg(feature = "cache")]
        let cache_key = if self.config.cache && self.cache_handler.is_some() && tools.is_empty() {
            Some(self.cache_key_for(&messages, options))
        } else {
            None
        };
        #[cfg(all(feature = "cache", not(feature = "inflight-dedup")))]
        if let (Some(key), Some(cache)) = (cache_key, self.cache_handler.as_ref())
            && let Some(raw_output) = cache.get_response(key).await?
        {
            debug!("lm response served from cache");
            return Ok(Self::response_from_cache(messages, raw_output));
        }

        #[cfg(feature = "inflight-dedup")]
        let pending = if let (Some(key), Some(cache)) = (cache_key, self.cache_handler.as_ref()) {
            match cache.shared_response(key).await? {
                SharedResponse::Cached(raw_output) => {
                    debug!("lm response shared from cache");
                    return Ok(Self::response_from_cache(messages, raw_output));
                }
                SharedResponse::Leader(pending) => Some(pending),
                SharedResponse::Independent => None,
            }
        } else {
            None
        };

        let system_prompt = messages.system_prompt();
        let chat_history = messages.to_rig_chat_history();
        let tool_definitions = tools.definitions();
        if matches!(self.client.as_deref(), Some(LMClient::Gemini(_)))
            && tool_definitions.iter().any(|tool| {
                !crate::typesys::json_schema::fits_inline_tool_schema_budget(&tool.parameters)
            })
        {
            return Err(crate::LmCallError::UnsupportedOption {
                option: "tool schema exceeds Gemini's bounded inline expansion",
            }
            .into());
        }
        trace!(
            conversation_messages = chat_history.len(),
            tool_definitions = tool_definitions.len(),
            "prepared completion request inputs"
        );

        let tool_choice = if !tool_definitions.is_empty() {
            Some(ToolChoice::Auto)
        } else {
            None
        };
        let prepared = PreparedCallOptions {
            json_mode: options.json_mode,
            output_schema,
        };

        // Execute the completion using enum dispatch (zero-cost abstraction),
        // retrying transient failures with backoff.
        let response = self
            .completion_with_retry(
                || {
                    self.build_completion_request(
                        &system_prompt,
                        &chat_history,
                        tool_definitions,
                        tool_choice.clone(),
                        prepared,
                    )
                },
                progress,
            )
            .await?;
        debug!(
            prompt_tokens = response.usage.input_tokens,
            completion_tokens = response.usage.output_tokens,
            total_tokens = response.usage.total_tokens,
            "lm completion received"
        );

        let first_usage = LmUsage::from(response.usage);

        let (output, chat, tool_calls, tool_executions) = match classify_choice(response.choice)? {
            ChoiceAction::Text(text) => {
                // Only a plain completion can populate the response cache.
                #[cfg(feature = "cache")]
                if let (Some(key), Some(cache)) = (cache_key, self.cache_handler.as_ref()) {
                    cache.insert_entry(
                        key,
                        CacheEntry {
                            prompt: messages.to_json().to_string(),
                            usage: progress.total_usage(),
                            raw_output: Some(text.clone()),
                        },
                    );
                    trace!("lm response cached");
                }
                #[cfg(feature = "inflight-dedup")]
                drop(pending);

                let output = Message::assistant(text);
                progress.events.push(SpanEvent::Exchange {
                    message: output.clone(),
                    usage: first_usage,
                });
                let mut chat = messages;
                chat.push_message(output.clone());
                (output, chat, Vec::new(), Vec::new())
            }
            ChoiceAction::ToolCalls {
                calls,
                assistant_text,
                full_content,
            } => {
                #[cfg(feature = "inflight-dedup")]
                if let Some(pending) = pending {
                    pending.finish_uncached();
                }
                match tool_loop_mode {
                    ToolLoopMode::Auto if !tools.is_empty() => {
                        debug!(count = calls.len(), "entering tool loop");
                        let ToolLoopResult {
                            message,
                            chat_history,
                            tool_calls,
                            tool_executions,
                        } = self
                            .execute_tool_loop(
                                ToolRound {
                                    calls,
                                    content: *full_content,
                                    usage: first_usage,
                                },
                                tools,
                                chat_history,
                                &system_prompt,
                                prepared,
                                progress,
                            )
                            .await?;
                        let mut chat = Self::chat_from_rig_history(system_prompt, chat_history);
                        chat.push_message(message.clone());
                        (message, chat, tool_calls, tool_executions)
                    }
                    ToolLoopMode::Auto => {
                        let names: Vec<_> =
                            calls.iter().map(|tc| tc.function.name.as_str()).collect();
                        warn!(?names, "tools requested but no tools available");
                        let output = Message::assistant(format!(
                            "Tool calls requested: {names:?}, but no tools available"
                        ));
                        let content = rig::OneOrMany::many(
                            calls.into_iter().map(AssistantContent::ToolCall),
                        )?;
                        let assistant =
                            Message::from(rig::message::Message::Assistant { id: None, content });
                        progress.events.push(SpanEvent::Exchange {
                            message: assistant.clone(),
                            usage: first_usage,
                        });
                        progress.events.push(SpanEvent::ChatMessage {
                            message: output.clone(),
                        });
                        let mut chat = messages;
                        chat.push_message(assistant);
                        chat.push_message(output.clone());
                        (output, chat, Vec::new(), Vec::new())
                    }
                    ToolLoopMode::CallerManaged => {
                        let assistant = Message::from(rig::message::Message::Assistant {
                            id: None,
                            content: *full_content,
                        });
                        progress.events.push(SpanEvent::Exchange {
                            message: assistant.clone(),
                            usage: first_usage,
                        });
                        let mut chat = messages;
                        chat.push_message(assistant);
                        (
                            Message::assistant(assistant_text.unwrap_or_default()),
                            chat,
                            calls,
                            Vec::new(),
                        )
                    }
                }
            }
        };
        let usage = progress.total_usage();
        debug!(
            tool_calls = tool_calls.len(),
            tool_executions = tool_executions.len(),
            total_tokens = usage.total_tokens,
            "lm call completed"
        );

        Ok(LMResponse {
            cache_hit: false,
            history: None,
            output,
            usage,
            chat,
            tool_calls,
            tool_executions,
            events: std::mem::take(&mut progress.events),
        })
    }

    #[cfg(feature = "cache")]
    fn response_from_cache(mut chat: Chat, raw_output: String) -> LMResponse {
        let output = Message::assistant(&raw_output);
        let usage = LmUsage::default();
        chat.push_message(output.clone());
        LMResponse {
            cache_hit: true,
            history: None,
            output: output.clone(),
            usage,
            chat,
            tool_calls: Vec::new(),
            tool_executions: Vec::new(),
            events: vec![SpanEvent::Exchange {
                message: output,
                usage,
            }],
        }
    }

    /// Returns up to `n` LM invocations, newest first, independent of caching.
    #[tracing::instrument(
        name = "dsrs.lm.inspect_history",
        level = "trace",
        skip(self),
        fields(n)
    )]
    pub async fn inspect_history(&self, n: usize) -> Vec<crate::HistoryEntry> {
        self.history.inspect(n)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use rig::OneOrMany;
    use rig::completion::AssistantContent;
    use rig::completion::ToolDefinition;
    use rig::tool::Tool;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn make_tool_call(name: &str) -> AssistantContent {
        AssistantContent::tool_call(
            format!("id_{name}"),
            name.to_string(),
            serde_json::json!({"arg": "val"}),
        )
    }

    fn make_reasoning(text: &str) -> AssistantContent {
        AssistantContent::reasoning(text)
    }

    fn make_text(text: &str) -> AssistantContent {
        AssistantContent::text(text)
    }

    #[tokio::test]
    async fn per_call_options_reach_mock_completion_request() {
        let model = TestCompletionModel::new([make_text("{}")]);
        let lm = test_lm_with_model(model.clone());
        let options = LMCallOptions {
            output_schema: Some(
                serde_json::json!({"type":"object", "properties":{}, "required":[], "additionalProperties":false}),
            ),
            force_output_schema: true,
            ..LMCallOptions::default()
        };
        lm.call_with_options(
            Chat::new(vec![Message::user("answer")]),
            Vec::new(),
            ToolLoopMode::CallerManaged,
            options,
        )
        .await
        .unwrap();
        let request = model.last_request().unwrap();
        assert!(request.output_schema.is_some());
        assert!(request.additional_params.is_none());
    }

    #[tokio::test]
    async fn json_mode_sets_additional_params_and_cache_keys_separate_formats() {
        let model = TestCompletionModel::new([make_text("{}")]);
        let lm = test_lm_with_model(model.clone());
        let options = LMCallOptions {
            json_mode: true,
            force_output_schema: true,
            ..LMCallOptions::default()
        };
        let chat = Chat::new(vec![Message::user("answer")]);
        lm.call_with_options(
            chat.clone(),
            Vec::new(),
            ToolLoopMode::CallerManaged,
            options.clone(),
        )
        .await
        .unwrap();
        let request = model.last_request().unwrap();
        assert_eq!(
            request.additional_params.unwrap()["response_format"]["type"],
            "json_object"
        );
        assert!(request.output_schema.is_none());
        #[cfg(feature = "cache")]
        assert_ne!(
            lm.cache_key_for(&chat, &LMCallOptions::default()),
            lm.cache_key_for(&chat, &options)
        );
    }

    #[tokio::test]
    async fn unsupported_client_omits_structured_request_options() {
        let model = TestCompletionModel::new([make_text("plain")]);
        let lm = test_lm_with_model(model.clone());
        let options = LMCallOptions {
            output_schema: Some(serde_json::json!({"type":"object"})),
            ..LMCallOptions::default()
        };
        lm.call_with_options(
            Chat::new(vec![Message::user("answer")]),
            Vec::new(),
            ToolLoopMode::CallerManaged,
            options,
        )
        .await
        .unwrap();
        let request = model.last_request().unwrap();
        assert!(request.output_schema.is_none());
        assert!(request.additional_params.is_none());
    }

    #[tokio::test]
    async fn provider_error_does_not_trigger_format_fallback_call() {
        let model = TestCompletionModel::default();
        let lm = test_lm_with_model(model.clone());
        let options = LMCallOptions {
            output_schema: Some(serde_json::json!({"type":"object"})),
            force_output_schema: true,
            ..LMCallOptions::default()
        };
        let result = lm
            .call_with_options(
                Chat::new(vec![Message::user("answer")]),
                Vec::new(),
                ToolLoopMode::CallerManaged,
                options,
            )
            .await;
        assert!(result.is_err());
        assert!(model.last_request().unwrap().output_schema.is_some());
    }

    #[test]
    fn retry_classifier_matches_transient_errors_only() {
        assert!(is_retryable_completion_error(
            &CompletionError::ProviderError("429 Too Many Requests".to_string())
        ));
        assert!(is_retryable_completion_error(
            &CompletionError::ProviderError("Anthropic: Overloaded".to_string())
        ));
        assert!(is_retryable_completion_error(
            &CompletionError::ProviderError("HTTP 503 service unavailable".to_string())
        ));
        assert!(!is_retryable_completion_error(
            &CompletionError::ProviderError("invalid api key".to_string())
        ));
        assert!(!is_retryable_completion_error(
            &CompletionError::ResponseError("test response queue is empty".to_string())
        ));
    }

    #[cfg(feature = "cache")]
    #[tokio::test]
    async fn cache_key_uses_sample_identity_and_effective_temperature() {
        let lm = LM {
            history: crate::HistoryBuffer::default(),
            config: LMConfig::default(),
            cache_handler: None,
            client: None,
            client_id: 0,
            custom_output_schema: false,
            scalar_output_schema: false,
        };
        let chat = Chat::new(vec![Message::user("same prompt")]);
        let first = crate::with_sampling_options(
            crate::SamplingOptions {
                id: Some("first".into()),
                temperature: Some(0.2),
            },
            async { lm.cache_key_for(&chat, &LMCallOptions::default()) },
        )
        .await;
        let same = crate::with_sampling_options(
            crate::SamplingOptions {
                id: Some("first".into()),
                temperature: Some(0.2),
            },
            async { lm.cache_key_for(&chat, &LMCallOptions::default()) },
        )
        .await;
        let distinct = crate::with_sampling_options(
            crate::SamplingOptions {
                id: Some("second".into()),
                temperature: Some(0.2),
            },
            async { lm.cache_key_for(&chat, &LMCallOptions::default()) },
        )
        .await;
        let temperature = crate::with_sampling_options(
            crate::SamplingOptions {
                id: Some("first".into()),
                temperature: Some(0.8),
            },
            async { lm.cache_key_for(&chat, &LMCallOptions::default()) },
        )
        .await;
        assert_eq!(first, same);
        assert_ne!(first, distinct);
        assert_ne!(first, temperature);
    }

    #[cfg(feature = "cache")]
    #[tokio::test]
    async fn cache_hit_reports_zero_usage_and_keeps_original_entry() {
        let cache = Arc::new(ResponseCache::memory_only_for_test().await);
        let lm = LM {
            history: crate::HistoryBuffer::default(),
            config: LMConfig {
                cache: true,
                ..LMConfig::default()
            },
            cache_handler: Some(cache.clone()),
            client: None,
            client_id: 0,
            custom_output_schema: false,
            scalar_output_schema: false,
        };
        let messages = Chat::new(vec![Message::user("cached prompt")]);
        let key = lm.cache_key_for(&messages, &LMCallOptions::default());
        let original = LmUsage {
            prompt_tokens: 12,
            completion_tokens: 3,
            total_tokens: 15,
        };
        cache.insert_entry(
            key,
            CacheEntry {
                prompt: messages.to_json().to_string(),
                usage: original,
                raw_output: Some("cached answer".into()),
            },
        );
        let response = lm
            .call_with_toolset(messages, &ToolSet::default(), ToolLoopMode::Auto)
            .await
            .unwrap();
        assert_eq!(response.usage.total_tokens, 0);
        assert!(response.events.iter().all(|event| match event {
            SpanEvent::Exchange { usage, .. } => usage.total_tokens == 0,
            _ => true,
        }));
        let entry = cache.get_entry(key).await.unwrap().unwrap();
        assert_eq!(entry.usage.total_tokens, original.total_tokens);
    }

    #[cfg(feature = "cache")]
    #[tokio::test]
    async fn equal_sample_identity_reuses_cache_but_distinct_identity_reaches_provider() {
        let model =
            TestCompletionModel::new([make_text("first answer"), make_text("second answer")]);
        let mut lm = test_lm_with_model(model.clone());
        lm.config.cache = true;
        lm.cache_handler = Some(Arc::new(ResponseCache::memory_only_for_test().await));
        let messages = Chat::new(vec![Message::user("same prompt")]);
        for (id, expected) in [
            ("same", "first answer"),
            ("same", "first answer"),
            ("different", "second answer"),
        ] {
            let response = crate::with_sampling_options(
                crate::SamplingOptions {
                    id: Some(id.into()),
                    temperature: None,
                },
                lm.call_with_toolset(messages.clone(), &ToolSet::default(), ToolLoopMode::Auto),
            )
            .await
            .unwrap();
            assert_eq!(response.output.text_content(), expected);
        }
        assert_eq!(model.call_count(), 2);
    }

    #[test]
    fn classify_text_only() {
        let choice = OneOrMany::one(make_text("hello"));
        match classify_choice(choice).unwrap() {
            ChoiceAction::Text(t) => assert_eq!(t, "hello"),
            ChoiceAction::ToolCalls { .. } => panic!("expected Text, got ToolCalls"),
        }
    }

    #[test]
    fn classify_single_tool_call() {
        let choice = OneOrMany::one(make_tool_call("search"));
        match classify_choice(choice).unwrap() {
            ChoiceAction::ToolCalls {
                calls,
                full_content,
                assistant_text,
            } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "search");
                assert_eq!(full_content.iter().count(), 1);
                assert!(assistant_text.is_none());
            }
            ChoiceAction::Text(_) => panic!("expected ToolCalls, got Text"),
        }
    }

    #[test]
    fn classify_reasoning_then_tool_call() {
        let choice = OneOrMany::many(vec![
            make_reasoning("thinking..."),
            make_tool_call("search"),
        ])
        .unwrap();

        match classify_choice(choice).unwrap() {
            ChoiceAction::ToolCalls {
                calls,
                full_content,
                assistant_text,
            } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "search");
                // full_content preserves both blocks
                assert_eq!(full_content.iter().count(), 2);
                assert!(assistant_text.is_none());
            }
            ChoiceAction::Text(_) => panic!("expected ToolCalls, got Text"),
        }
    }

    #[test]
    fn classify_reasoning_then_text() {
        let choice = OneOrMany::many(vec![
            make_reasoning("let me think"),
            make_text("the answer is 42"),
        ])
        .unwrap();

        match classify_choice(choice).unwrap() {
            ChoiceAction::Text(t) => assert_eq!(t, "the answer is 42"),
            ChoiceAction::ToolCalls { .. } => panic!("expected Text, got ToolCalls"),
        }
    }

    #[test]
    fn classify_reasoning_only_fallback() {
        let choice = OneOrMany::one(make_reasoning("just thinking"));
        match classify_choice(choice).unwrap() {
            ChoiceAction::Text(t) => assert_eq!(t, "just thinking"),
            ChoiceAction::ToolCalls { .. } => panic!("expected Text, got ToolCalls"),
        }
    }

    #[test]
    fn classify_tool_call_wins_over_text() {
        let choice =
            OneOrMany::many(vec![make_text("some text"), make_tool_call("search")]).unwrap();

        match classify_choice(choice).unwrap() {
            ChoiceAction::ToolCalls {
                calls,
                assistant_text,
                ..
            } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "search");
                assert_eq!(assistant_text.as_deref(), Some("some text"));
            }
            ChoiceAction::Text(_) => panic!("expected ToolCalls, got Text"),
        }
    }

    #[test]
    fn classify_multiple_tool_calls() {
        let choice = OneOrMany::many(vec![
            make_reasoning("planning"),
            make_tool_call("search"),
            make_tool_call("calculate"),
        ])
        .unwrap();

        match classify_choice(choice).unwrap() {
            ChoiceAction::ToolCalls {
                calls,
                full_content,
                assistant_text,
            } => {
                assert_eq!(calls.len(), 2);
                assert_eq!(calls[0].function.name, "search");
                assert_eq!(calls[1].function.name, "calculate");
                assert_eq!(full_content.iter().count(), 3);
                assert!(assistant_text.is_none());
            }
            ChoiceAction::Text(_) => panic!("expected ToolCalls, got Text"),
        }
    }

    #[test]
    fn classify_image_only_returns_error() {
        let choice = OneOrMany::one(AssistantContent::Image(
            rig::completion::message::Image::default(),
        ));
        let error = classify_choice(choice)
            .err()
            .expect("image output must fail");
        assert!(error.to_string().contains("Image responses"));
        assert!(matches!(
            error.downcast_ref::<crate::LmCallError>(),
            Some(crate::LmCallError::UnsupportedContent)
        ));
    }

    #[test]
    fn classify_image_with_text_preserves_text() {
        let choice = OneOrMany::many(vec![
            AssistantContent::Image(rig::completion::message::Image::default()),
            make_text("caption"),
        ])
        .unwrap();
        match classify_choice(choice).unwrap() {
            ChoiceAction::Text(text) => assert_eq!(text, "caption"),
            _ => panic!("expected caption"),
        }
    }

    #[tokio::test]
    async fn image_response_errors_propagate_from_initial_and_tool_loop_completions() {
        for after_tool in [false, true] {
            let mut responses = Vec::new();
            if after_tool {
                responses.push(make_tool_call("counter"));
            }
            responses.push(AssistantContent::Image(
                rig::completion::message::Image::default(),
            ));
            let lm = test_lm_with_model(TestCompletionModel::new(responses));
            let calls = Arc::new(AtomicUsize::new(0));
            let tools: Vec<Arc<dyn ToolDyn>> = if after_tool {
                vec![Arc::new(CountingTool {
                    calls: Arc::clone(&calls),
                })]
            } else {
                vec![]
            };
            let error = lm
                .call(Chat::new(vec![Message::user("hello")]), tools)
                .await
                .expect_err("image response must fail");
            assert!(error.to_string().contains("Image responses"));
            assert_eq!(calls.load(Ordering::SeqCst), usize::from(after_tool));
        }
    }

    #[derive(Clone)]
    struct CountingTool {
        calls: Arc<AtomicUsize>,
    }

    #[derive(Debug)]
    struct CountingToolError;

    impl std::fmt::Display for CountingToolError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "counting tool error")
        }
    }

    impl std::error::Error for CountingToolError {}

    impl Tool for CountingTool {
        const NAME: &'static str = "counter";
        type Error = CountingToolError;
        type Args = serde_json::Value;
        type Output = String;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            ToolDefinition {
                name: Self::NAME.to_string(),
                description: "counter tool".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "additionalProperties": true
                }),
            }
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok("counted".to_string())
        }
    }

    pub(crate) fn test_lm_with_model(model: TestCompletionModel) -> LM {
        LM {
            history: crate::HistoryBuffer::default(),
            config: LMConfig {
                native_tools: true,
                base_url: None,
                api_key: None,
                model: "openai:gpt-4o-mini".to_string(),
                temperature: 0.0,
                max_tokens: 128,
                max_tool_iterations: 4,
                max_retries: 0,
                retry_base_delay_ms: 1,
                cache: false,
            },
            #[cfg(feature = "cache")]
            cache_handler: None,
            client: Some(Arc::new(LMClient::Test(model))),
            client_id: LM::next_client_id(),
            custom_output_schema: false,
            scalar_output_schema: false,
        }
    }

    #[rstest::rstest]
    #[case::provider_error(false)]
    #[case::cancellation(true)]
    #[tokio::test]
    async fn unfinished_auto_tool_loop_keeps_completed_round(#[case] cancel: bool) {
        struct StallAfterFirst {
            model: TestCompletionModel,
            started: Arc<tokio::sync::Notify>,
        }

        impl CompletionProvider for StallAfterFirst {
            async fn completion(
                &self,
                request: CompletionRequest,
            ) -> Result<CompletionResponse<()>, CompletionError> {
                if self.model.call_count() == 0 {
                    return self.model.completion(request).await;
                }
                self.started.notify_one();
                std::future::pending().await
            }
        }

        let model = TestCompletionModel::new([make_tool_call("counter")]);
        model.set_usage(rig::completion::Usage {
            input_tokens: 15,
            output_tokens: 5,
            total_tokens: 20,
            ..rig::completion::Usage::new()
        });
        let mut lm = test_lm_with_model(model.clone());
        lm.history.set_capacity(4);
        let started = Arc::new(tokio::sync::Notify::new());
        if cancel {
            lm.client = Some(Arc::new(LMClient::custom(StallAfterFirst {
                model,
                started: Arc::clone(&started),
            })));
        }

        let calls = Arc::new(AtomicUsize::new(0));
        let tools: Vec<Arc<dyn ToolDyn>> = vec![Arc::new(CountingTool {
            calls: Arc::clone(&calls),
        })];
        let chat = Chat::new(vec![Message::user("Use the counter")]);
        let task = tokio::spawn({
            let lm = lm.clone();
            async move { lm.call(chat, tools).await }
        });
        let error_kind = if cancel {
            tokio::time::timeout(Duration::from_secs(5), started.notified())
                .await
                .expect("the second provider call started");
            task.abort();
            assert!(task.await.unwrap_err().is_cancelled());
            crate::trace::SpanErrorKind::Cancelled
        } else {
            assert!(task.await.unwrap().is_err());
            crate::trace::SpanErrorKind::Lm
        };

        assert_eq!(calls.load(Ordering::Relaxed), 1);
        let history = lm.inspect_history(4).await;
        assert_eq!(history.len(), 1);
        let entry = &history[0];
        assert_eq!(entry.error.as_ref().unwrap().kind, error_kind);
        assert_eq!(entry.usage.unwrap().total_tokens, 20);
        assert!(matches!(
            entry.events.as_slice(),
            [SpanEvent::Exchange { .. }, SpanEvent::ToolRun { .. }]
        ));
    }

    #[cfg(feature = "inflight-dedup")]
    async fn cached_test_lm(model: TestCompletionModel) -> LM {
        let mut lm = test_lm_with_model(model);
        lm.config.cache = true;
        lm.cache_handler = Some(Arc::new(ResponseCache::memory_only_for_test().await));
        lm
    }

    #[cfg(feature = "inflight-dedup")]
    #[tokio::test]
    async fn identical_call_reuses_inflight_response_with_zero_usage() {
        let model = TestCompletionModel::new([]);
        let lm = cached_test_lm(model.clone()).await;
        let chat = Chat::new(vec![Message::user("hi")]);
        let key = lm.cache_key_for(&chat, &LMCallOptions::default());
        let cache = lm.cache_handler.as_ref().unwrap();
        let InflightResponse::Leader(pending) = cache.register_inflight(key) else {
            panic!("expected to own the in-flight slot");
        };
        let mut waiter = tokio::spawn({
            let lm = lm.clone();
            let chat = chat.clone();
            async move { lm.call(chat, vec![]).await }
        });
        assert!(
            tokio::time::timeout(Duration::from_millis(50), &mut waiter)
                .await
                .is_err()
        );

        cache.insert_entry(
            key,
            CacheEntry {
                prompt: chat.to_json().to_string(),
                usage: LmUsage::default(),
                raw_output: Some("shared".into()),
            },
        );
        drop(pending);

        let response = tokio::time::timeout(Duration::from_secs(5), waiter)
            .await
            .expect("waiter did not finish")
            .unwrap()
            .unwrap();
        assert_eq!(response.output.content(), "shared");
        assert!(response.cache_hit);
        assert_eq!(response.usage.total_tokens, 0);
        assert!(model.last_request().is_none());
    }

    #[cfg(feature = "inflight-dedup")]
    struct DelayedFirstFailure {
        calls: Arc<AtomicUsize>,
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
        recovery_started: Arc<tokio::sync::Notify>,
        recover: Arc<tokio::sync::Notify>,
        model: TestCompletionModel,
    }

    #[cfg(feature = "inflight-dedup")]
    impl CompletionProvider for DelayedFirstFailure {
        async fn completion(
            &self,
            request: CompletionRequest,
        ) -> Result<CompletionResponse<()>, CompletionError> {
            if self.calls.fetch_add(1, Ordering::Relaxed) == 0 {
                self.started.notify_one();
                self.release.notified().await;
                return Err(CompletionError::ResponseError(
                    "in-flight leader failed".into(),
                ));
            }
            self.recovery_started.notify_one();
            self.recover.notified().await;
            self.model.completion(request).await
        }
    }

    #[cfg(feature = "inflight-dedup")]
    #[rstest::rstest]
    #[case::failure(false)]
    #[case::cancellation(true)]
    #[tokio::test]
    async fn identical_call_falls_back_when_inflight_produces_no_response(#[case] cancel: bool) {
        const WAITERS: usize = 8;
        let model = TestCompletionModel::new([make_text("fresh")]);
        let calls = Arc::new(AtomicUsize::new(0));
        let started = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let recovery_started = Arc::new(tokio::sync::Notify::new());
        let recover = Arc::new(tokio::sync::Notify::new());
        let mut lm = cached_test_lm(model.clone()).await;
        lm.client = Some(Arc::new(LMClient::custom(DelayedFirstFailure {
            calls: Arc::clone(&calls),
            started: started.clone(),
            release: release.clone(),
            recovery_started: Arc::clone(&recovery_started),
            recover: Arc::clone(&recover),
            model: model.clone(),
        })));
        let chat = Chat::new(vec![Message::user("hi")]);
        let key = lm.cache_key_for(&chat, &LMCallOptions::default());
        let cache = lm.cache_handler.as_ref().unwrap();
        let leader = tokio::spawn({
            let lm = lm.clone();
            let chat = chat.clone();
            async move { lm.call(chat, vec![]).await }
        });
        tokio::time::timeout(Duration::from_secs(5), started.notified())
            .await
            .expect("leader did not reach the provider");
        let mut waiters = std::pin::pin!(futures::future::join_all(
            (0..WAITERS).map(|_| lm.call(chat.clone(), vec![]))
        ));
        cache
            .drive_until_waiting(key, waiters.as_mut(), WAITERS)
            .await;

        if cancel {
            leader.abort();
            assert!(leader.await.unwrap_err().is_cancelled());
        } else {
            release.notify_one();
            assert!(leader.await.unwrap().is_err());
        }

        cache
            .drive_until_waiting(key, waiters.as_mut(), WAITERS - 1)
            .await;
        tokio::select! {
            () = recovery_started.notified() => {}
            _ = &mut waiters => panic!("waiters completed before recovery was released"),
            _ = tokio::time::sleep(Duration::from_secs(10)) => panic!("recovery did not start"),
        }
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        recover.notify_waiters();
        let responses = tokio::time::timeout(Duration::from_secs(5), waiters)
            .await
            .expect("waiters did not finish");
        let mut cache_hits = 0;
        for response in responses {
            let response = response.unwrap();
            assert_eq!(response.output.content(), "fresh");
            if response.cache_hit {
                assert_eq!(response.usage.total_tokens, 0);
                cache_hits += 1;
            }
        }
        assert_eq!(cache_hits, WAITERS - 1);
        assert_eq!(model.call_count(), 1);
    }

    #[tokio::test]
    async fn call_with_caller_managed_mode_returns_tool_calls_without_executing() {
        let model = TestCompletionModel::new([make_tool_call("counter")]);
        let lm = test_lm_with_model(model);

        let call_count = Arc::new(AtomicUsize::new(0));
        let tools: Vec<Arc<dyn ToolDyn>> = vec![Arc::new(CountingTool {
            calls: Arc::clone(&call_count),
        })];

        let chat = Chat::new(vec![Message::user("Use the counter tool")]);
        let response = lm
            .call_with_tool_loop_mode(chat, tools, ToolLoopMode::CallerManaged)
            .await
            .expect("caller-managed call should succeed");

        assert_eq!(response.tool_calls.len(), 1);
        assert!(response.tool_executions.is_empty());
        assert_eq!(call_count.load(Ordering::SeqCst), 0);
        assert_eq!(response.output.content(), "");
        assert_eq!(response.chat.len(), 2);
        assert!(response.chat.messages[1].has_tool_calls());
    }

    #[tokio::test]
    async fn call_default_auto_mode_executes_tool_loop() {
        let model = TestCompletionModel::new([make_tool_call("counter"), make_text("done")]);
        let lm = test_lm_with_model(model);

        let call_count = Arc::new(AtomicUsize::new(0));
        let tools: Vec<Arc<dyn ToolDyn>> = vec![Arc::new(CountingTool {
            calls: Arc::clone(&call_count),
        })];

        let chat = Chat::new(vec![Message::user("Use the counter tool")]);
        let response = lm
            .call(chat, tools)
            .await
            .expect("auto call should succeed");

        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_executions.len(), 1);
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
        assert_eq!(response.output.content(), "done");
        assert_eq!(response.chat.len(), 4);
        assert!(response.chat.messages[1].has_tool_calls());
        assert!(response.chat.messages[2].has_tool_results());
        assert_eq!(response.chat.messages[3].role, Role::Assistant);

        // Ordered per-round-trip record: tool-call exchange, tool run, final text.
        assert_eq!(response.events.len(), 3);
        assert!(
            matches!(&response.events[0], SpanEvent::Exchange { message, .. } if message.has_tool_calls())
        );
        // rig's ToolDyn::call JSON-encodes the output, hence contains not equals.
        assert!(
            matches!(&response.events[1], SpanEvent::ToolRun { name, result, .. } if name == "counter" && result.contains("counted"))
        );
        assert!(
            matches!(&response.events[2], SpanEvent::Exchange { message, .. } if message.content() == "done")
        );
    }

    #[tokio::test]
    async fn plain_text_call_records_single_exchange_event() {
        let model = TestCompletionModel::new([make_text("hello")]);
        let lm = test_lm_with_model(model);

        let chat = Chat::new(vec![Message::user("hi")]);
        let response = lm.call(chat, vec![]).await.expect("call should succeed");

        assert!(
            matches!(response.events.as_slice(), [SpanEvent::Exchange { message, .. }] if message.content() == "hello")
        );
    }
}
