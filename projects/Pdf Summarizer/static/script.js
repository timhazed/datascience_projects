let lightMode = true;
let isFirstMessage = true;
const responses = [];
const baseUrl = window.location.origin;

// Show bot loading animation
async function showBotLoadingAnimation() {
  await sleep(200);
  $(".loading-animation")[1].style.display = "inline-block";
  document.getElementById('send-button').disabled = true;
}

// Hide bot loading animation
function hideBotLoadingAnimation() {
  $(".loading-animation")[1].style.display = "none";
  if (!isFirstMessage) {
    document.getElementById('send-button').disabled = false;
  }
}

// Sleep function for delays
const sleep = (time) => new Promise((resolve) => setTimeout(resolve, time));

// Scroll to the bottom of the chat window
const scrollToBottom = () => {
  $("#chat-window").animate({
    scrollTop: $("#chat-window")[0].scrollHeight,
  });
};

// Clean and sanitize user input
const cleanTextInput = (value) => {
  return value
    .trim()
    .replace(/[\n\t]/g, "")
    .replace(/<[^>]*>/g, "")
    .replace(/[<>&;]/g, "");
};

// Handle file upload
const handleFileUpload = async (file) => {
  await showBotLoadingAnimation();
  const formData = new FormData();
  formData.append("file", file);

  let response = await fetch(baseUrl + "/process-document", {
    method: "POST",
    headers: { Accept: "application/json" },
    body: formData,
  });

  response = await response.json();
  renderBotResponse(response);
};

// Populate user message in the chat window
const populateUserMessage = (userMessage) => {
  $("#message-input").val("");

  $("#message-list").append(
    `<div class='message-line my-text'><div class='message-box my-text${
      !lightMode ? " dark" : ""
    }'><div class='me'>${userMessage}</div></div></div>`
  );

  scrollToBottom();
};

// Render bot response in the chat window
const renderBotResponse = (response) => {
  responses.push(response);
  hideBotLoadingAnimation();

  $("#message-list").append(
    `<div class='message-line'><div class='message-box${!lightMode ? " dark" : ""}'>${response.botResponse.trim()}</div></div>`
  );

  scrollToBottom();
};

// Process user message
const processUserMessage = async (userMessage) => {
  let response = await fetch(baseUrl + "/process-message", {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify({ userMessage: userMessage }),
  });
  response = await response.json();
  return response;
};

// Populate bot response for the first message
const populateBotResponse = async (userMessage = '') => {
  await showBotLoadingAnimation();
  let response;

  if (isFirstMessage) {
    response = { botResponse: "I'm your PDF summarizer, ready to answer any questions regarding your data. Please upload a PDF file to analyze." };

    $("#message-list").append(
      `<input type="file" id="file-upload" accept=".pdf" hidden>
      <button id="upload-button" class="btn btn-primary btn-sm">Upload PDF</button>`
    );

    $("#upload-button").on("click", function () {
      $("#file-upload").click();
    });

    $("#file-upload").on("change", async function () {
      const file = this.files[0];
      if (file) {
        await handleFileUpload(file);
      }
    });

    isFirstMessage = false;
  } else {
    response = await processUserMessage(userMessage);
  }

  renderBotResponse(response);
};

// Start conversation
populateBotResponse();

$(document).ready(function () {
  $("#message-input").keyup(function (event) {
    let inputVal = cleanTextInput($("#message-input").val());
    if (event.keyCode === 13 && inputVal !== "") {
      const message = inputVal;
      populateUserMessage(message);
      populateBotResponse(message);
    }
  });

  $("#send-button").click(async function () {
    const message = cleanTextInput($("#message-input").val());
    populateUserMessage(message);
    populateBotResponse(message);
  });

  $("#reset-button").click(function () {
    $("#message-list").empty();
    responses.length = 0;
    isFirstMessage = true;
    populateBotResponse();
  });

  $("#light-dark-mode-switch").change(function () {
    $("body").toggleClass("dark-mode");
    $(".message-box").toggleClass("dark");
    lightMode = !lightMode;
  });
});
