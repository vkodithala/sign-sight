# Sign-Speak
![image](https://github.com/user-attachments/assets/338b1dca-9bc8-48a8-a3a4-5959751999a4)

## Inspiration

The four of us had several shared experiences with the deaf community during grade school and were alarmed by the barriers they face in everyday conversation. Deaf people are often relegated to communicating via ASL, which very few outside of their close friends and family understand. In many situations, this confines their conversations to text, rather than the natural back-and-forth dialogue we all appreciate.

What’s particularly concerning is that most commercially available ASL translation devices aren't practical for everyday conversation. Most are worn by readers rather than signers, suggesting that the average person has a means of translating ASL to text when interacting with a signer. Others are expensive and bulky ASL detection gloves that interfere with the daily activities of signers.

We wondered if we could do better. We asked ourselves how we could leverage recent advancements in computer vision, [hand landmark detection](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer), deep learning, and [edge computing](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/) to develop a wearable device for the deaf that translates ASL signs into spoken audio in real-time, breaking down the barriers to communication that currently exist between the deaf community and the rest of the world.

## What it does

Sign-Speak is a pair of smartglasses that allow deaf users to translate their ASL signs into spoken audio in real-time. 

Most existing solutions focus on translation from the reader's perspective, and to our knowledge there isn't anyone working on translating ASL from the signer's perspective. Evidence for this is the lack of labelled data for back-of-hand ASL signs, which is a challenge that we overcame early in the project (see "Challenges we ran into").

Built with MediaPipe's hand-tracking neural network and hosted on NVIDIA's Jetson Nano, the glasses extract precise hand landmarks, process them through a custom neural network trained on our custom ASL dataset focused only on backside hand movements, and deliver instant voice translations via an attached speaker. Our CV and speech-to-text models are hosted completely **on device**, making Sign-Speak a true Edge AI product pushing the frontiers of accessibility tech. 

Our goal is to empower the deaf community to communicate freely and create a world where every conversation transcends the barriers of sound.

## How we built it

There weren't any available datasets for back-of-hand ASL signs, so the first step in training our model was to create one. To do so, we developed some [scripts](https://github.com/vkodithala/sign-speak/tree/main/data-collection) that took and labelled nearly 6,000 pictures of our own hands while we were signing the 26 letters of the ASL alphabet (along with some useful stopwords). We then used Google's MediaPipe models to translate these pictures into 21 coordinates representing the joints in the model's hands ("landmarks"). Our last step was to apply random scaling, rotation, and Gaussian noise to the extracted landmarks and train a feed-forward neural network that classifies ASL signs from inputted hand landmark data.

The brains of our operation rely on the small but powerful NVIDIA Jetson Orin Nano upon which our computer vision model runs, allowing our users to use SignSight wherever they want. We packaged our image classification model into a .onnx file and uploaded it onto our Jetson to run inference. The Jetson itself is connected to a webcam that captures input, passes it through the model to detect what letters the user is signing in succession, and then uses [Piper](https://github.com/rhasspy/piper) for on-device text-to-speech translation, sending the audio to a miniature Bluetooth speaker connected to the frames.

We 3D-printed our glasses based on custom CAD designs to allow for the mounting of a webcam in the temple area, and fashioned an elastic band around the back of the frames with a Bluetooth speaker to enable the sound output of the TTS translation. Users are meant to carry the Jetson Nano (which is housed in a custom 3D-printed case) in their pocket, allowing them to use Sign-Speak's technology anywhere on Earth.

## Challenges we ran into

*No existing model or dataset currently focuses on first-person American Sign Language (ASL).* Most available ASL datasets and computer vision models concentrate solely on capturing signs from the front, as seen by an observer, rather than from the perspective of the signer. Initially, we considered utilizing synthetic data from vision models. However, we discovered that these models had not been trained on sufficient footage of back-of-hand ASL signs, which hindered the generation of high-quality synthetic data for our needs. 

As a result, we faced the challenge of developing our dataset and model from scratch. This involved taking thousands of photographs of ASL signs from a back-of-hand perspective. To do this, we implemented MediaLabs' Hand Landmark Detection model, which was effective at estimating finger landmarks, even when joints were obscured from the camera's view. We used this model to convert our images into hand landmarks and used these as inputs to our model.

*Model overfitting and confusion among similar letters.* One significant challenge in developing a robust dataset for back-of-hand ASL detection lies in the complexity of the task. Many signs appear remarkably similar from this perspective, leading to confusion and decreased confidence in our model for certain words. To address this, we utilized hand landmarks instead of relying solely on raw image pixels. This approach allowed MediaPipe to effectively infer the positions of obscured fingers, enhancing our model’s accuracy. Additionally, we faced issues with overfitting due to the custom dataset we created. To mitigate this, we carefully managed the number of training epochs and introduced random noise to improve our model's generalization capabilities.

## Accomplishments that we're proud of
- We created the first ever dataset and computer vision model for the the ASL alphabet from the speaker's perspective.
- We developed and trained a custom feed-forward neural network that performed well on classifying the 26 letters of the ASL alphabet.
- We created and 3D-printed custom CAD designs for a pair of smartglasses built to mount a webcam and a case for our Jetson Nano.
- We configured an OLED display to show the current word that the user is spelling before it's outputted from text-to-speech.

## What we learned
While creating SignSight, we learned about developing deep learning models on constrained resources and the tradeoffs associated with developing on small computers. While developing a seamless user experience, we realized that effective assistive technology must blend into daily life without adding complexity. It took thoughtful design choices to decide how a Jetson Nano, speaker, and camera would fit into the form factor of wearable glasses.

## What's next for Sign-Speak
- **Two-way communication.** That is, in addition to a hearing person being able to hear a deaf person, a deaf person should be able to understand a non-signing person by translating speech to text. The text would be displayed in a small OLED display on the lens of SignSight in real time as the microphone picks up on speech. During TreeHacks, we spent time trying to enable two-way communication using these glasses. However, due to the constraints of the specific OLED display we got on our hands on, we weren't able to find any libraries that were compatible for our use case. We even tried to create our own library and modify existing ones—work that will continue past TreeHacks. We aim to finish SignSight's speech to text capability to truly ensure no voice is unheard.
- **Adding ASL word recognition.** ASL word recognition is a uniquely challenging task because the number of things that you can sign grows far beyond just 26 letters and there are an infinite amount of configurations that you can produce, some of which are in the signer's line of sight and others that aren't. Though we couldn't implement it during this hackathon, our project would be greatly enhanced by training a model capable of recognizing entire words from the ASL vocabulary via a combination of vision models and additional sensors.

## License 📜
Copyright 2025 ©Varoon Kodithala, Mehul Rao, Arnav Patidar, Vineeth Sendilraj

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
