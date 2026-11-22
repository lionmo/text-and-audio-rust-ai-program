use std::io::{Write, stdin, stdout};

use any_tts::{
    load_model,
    ModelType,
    SynthesisRequest,
    TtsConfig,
};
use futures::StreamExt;
use mistralrs::{GgufModelBuilder, Response, TextMessageRole, TextMessages};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tex_model = GgufModelBuilder::new(
        "bartowski/Llama-3.2-1B-Instruct-GGUF",
        vec!["Llama-3.2-1B-Instruct-Q4_K_M.gguf"],
    )
    .with_logging()
    .build()
    .await?;

    // Keep full conversation history across turns
    let mut messages = TextMessages::new().add_message(
        TextMessageRole::System,
        "You are a creative storyteller. Write plain text only with no markdown formatting like ** or # so it reads cleanly aloud.",
    );
    let mut last_response = String::new();

    println!("Discuss your story with the AI. Type 'yes' when you want to turn the latest story into audio.\n");

    let final_response = loop {
        print!("You: ");
        stdout().flush()?;

        let mut user_input = String::new();
        stdin().read_line(&mut user_input)?;
        let input = user_input.trim();

        if input.is_empty() {
            continue;
        }

        // Exit conversation and keep the latest AI story when user types "yes"
        if input.eq_ignore_ascii_case("yes") {
            if last_response.is_empty() {
                println!("Generate a story first before typing 'yes'.\n");
                continue;
            }
            break last_response;
        }

        messages = messages.add_message(TextMessageRole::User, input);
        let mut stream = tex_model.stream_chat_request(messages.clone()).await?;

        print!("AI: ");
        stdout().flush()?;

        let mut response = String::new();
        while let Some(chunk) = stream.next().await {
            match chunk {
                Response::Chunk(c) => {
                    if let Some(delta) = c.choices.first().and_then(|ch| ch.delta.content.as_deref()) {
                        print!("{}", delta);
                        stdout().flush()?;
                        response.push_str(delta);
                    }
                }
                Response::Done(_) => break,
                _ => {}
            }
        }
        println!("\n");

        // Save AI reply into history so the next turn remembers it
        messages = messages.add_message(TextMessageRole::Assistant, &response);
        last_response = response;
    };

    println!("Generating output.wav...");
    let model = load_model(
        TtsConfig::new(ModelType::Kokoro)
            //.with_model_path("./models/Kokoro-82M")
    )?;

    let request = SynthesisRequest::new(final_response)
        .with_language("en")
        .with_voice("af_heart")
        .with_speed(1.1);

    let audio = model.synthesize(&request)?;
    audio.save_wav("output.wav")?;
    println!("Saved output.wav!");

    Ok(())
}