let loader;
let submitButton;
let searchBar;
let results;
const test = false;

document.addEventListener("DOMContentLoaded", function () {
    loader = document.getElementById("loading-screen");
    loader.style.display = 'none';

    searchBar = document.getElementById('searchBar');
    results = document.getElementById('results');

    submitButton = document.getElementsByClassName("button-2")[0];
    submitButton.addEventListener('click', function () {
        textToImage().then((data) => console.log);
    });
});

const server_port = 5000;
const server_addr = "127.0.0.1";

async function textToImage() {
    const inputValue = searchBar.value;
    if (!inputValue) {
        alert("Error: Query Required!")
        return;
    }

    while(results.firstChild){
        results.removeChild(results.firstChild);
    }

    loader.style.display = 'block';
    submitButton.disabled = true;
    searchBar.disabled = true;

	const url = test ? `http://${server_addr}:${server_port}`: 
        'https://notebooksa.jarvislabs.ai/fd4NNqsipP_wrGj5q1wb0U0rNdMSjq2hQR46e4Jx9jc68BhSnkvBV8ofx1ZPQMMs'
    
    let res = await fetch(`${url}/image?` +
        new URLSearchParams({ q: inputValue }),
        {
            method: 'GET',
            keepalive: true
        });
  
    let responseText = '';
    if (res.status == 200) {
        const imageBlob = await res.blob()
        const imageObjectURL = URL.createObjectURL(imageBlob);

        const image = document.createElement('img')
        image.src = imageObjectURL

        results.append(image)
        responseText = 'Success!!!';
    } else {
        responseText = `HTTP error: ${res.status}`;
    }

    loader.style.display = 'none';
    submitButton.disabled = false;
    searchBar.disabled = false;

    console.log(responseText);
}

async function fetchWithTimeout(resource, options = {}) {
    const { timeout = 8000 } = options;
    
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);
    const response = await fetch(resource, {
      ...options,
      signal: controller.signal  
    });
    clearTimeout(id);
    return response;
}