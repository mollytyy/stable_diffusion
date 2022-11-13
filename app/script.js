let loader;
let submitButton;
let searchBar;
let results;

document.addEventListener("DOMContentLoaded", function () {
    loader = document.getElementById("loader");
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

    // let res = await fetch(`http://${server_addr}:${server_port}/image?` +
    //     new URLSearchParams({ q: inputValue }),
    //     {
    //         method: 'GET',
    //         mode: 'no-cors',
    //         headers: {
    //             "Accept": "image/*"
    //         }
    //     });

    let res = await fetch(`https://picsum.photos/200/300`,
        {
            method: 'GET'
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