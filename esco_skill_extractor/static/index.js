/**
 * This function is called when the form is submitted to ask the server to extract
 * the skills from the text. Then it updates the DOM with the extracted skills.
 * @param {Event} event the event object of the form submission.
 * @param {string} endpoint the endpoint to send the data to.
 */
async function submitForm(event, endpoint) {
  event.preventDefault();

  const text = document.getElementById("text").value;
  const output = document.getElementById("output");

  const response = await fetch(endpoint, {
    method: "POST",
    body: JSON.stringify([text]),
    headers: {
      "Content-Type": "application/json",
    },
  });

  const data = await response.json();
  const skillsStr = data[0].join(", ");

  output.innerHTML = `
    <button onclick="copyToClipboard('${skillsStr}')" id="copyButton"> Copy CSV </button>
    <ul>
      ${data[0]
        .map(
          (skill) =>
            `<li> 
              <a href="${skill}"> ${skill} </a> 
            </li>`
        )
        .join("")}
    </ul>`;
}

async function copyToClipboard(text) {
  try {
    const button = document.getElementById("copyButton");
    await navigator.clipboard.writeText(text);
    button.innerHTML = "Copied!";
    setTimeout(() => (button.innerHTML = "Copy CSV"), 1000);
  } catch (err) {
    console.error("Failed to copy: ", err);
  }
}

// Submit the form with ctrl+enter when the textarea is focused.
document.addEventListener("DOMContentLoaded", () =>
  document.getElementById("text").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.ctrlKey) submitForm(event, "/extract");
  })
);
