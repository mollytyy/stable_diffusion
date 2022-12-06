let loader;
let submitButton;
let searchBar;
let results;
let promptButton;
const promptList = [
    `Driving in the USA can be confusing not just because of all the rules and laws that drivers must follow, but also because of driving customs. Many people in the USA are really dependent on their cars to get to work and school. In fact most American workers spend an hour driving to work each day. In order to drive in the USA, you have to go to your local Department of Motor Vehicles first, and take a written test to get your learner's permit. If you pass this test, you can practice driving so you can pass a road test and get your license.`,
    "In USA cities like New York City and Los Angeles, many people live in small apartments. Despite not having homes with big yards, some apartment dwellers still seek the companionship that domesticated animals like dogs and cats offer. Some people feel that having a pet, even in a small space, is good for teaching children responsibility.",
    "Monday was the day of the big test. Luella had three days left to study. She knew she had to study for at least three hours per day in the next couple of days if she hoped to get the grade she wanted. Luella needed to score at least a ninety percent in her test in order to pass her Algebra class with an A.",
    "Surfing was Jeffrey's favorite pastime. Every weekend Jeffrey and his best friend Chad would head to Santa Monica Beach, where they would surf big waves. California was the perfect place to live if you were a surfer. The sunny sky and nice temperature made it an ideal place for surfing. Best of all, California mostly maintains this kind of nice weather throughout the year. Jeffrey and Chad were grateful to be surfers in California and not in Iceland. While their other friends were skateboarding and rollerblading, Jeffrey and Chad were at the beach surfing the day away.",
    "Jose Luis was taking the train to work one day when he noticed a woman with a big German Shepard getting on the train. This made him nervous. What if that dog attacked someone? Jose Luis didn't know that people could bring pets with them onto trains. It seemed unsafe. He stepped closer to the dog. The German Shepard seemed friendly enough. Just as he put out his hand to pet the dog, the woman stopped him."
];
const test = false;

document.addEventListener("DOMContentLoaded", function () {
    loader = document.getElementById("loading-screen");
    loader.style.display = 'none';

    searchBar = document.getElementById('searchBar');
    results = document.getElementById('results');

    submitButton = document.getElementById("search-btn");

    promptButton = document.getElementById("generate-btn");

    submitButton.addEventListener('click', function () {
        while(results.firstChild){
            results.removeChild(results.firstChild);
        }
        
        const splitInput = searchBar.value
            .split('.')
            .filter(s => s.length >= 2)
            .map(s => s.trim().replace(/(?:\r\n|\r|\n)/g, ''))

        generateImages(splitInput).then();
    });

    promptButton.addEventListener('click', function() {
        searchBar.value = promptList[Math.floor((Math.random()*promptList.length))];        
    });
});

async function generateImages(inputList) {
    for (const idx in inputList) {
        await textToImage(inputList[idx], idx);
    }
}

const server_port = 5000;
const server_addr = "127.0.0.1";

async function textToImage(inputValue, idx) {

    if (!inputValue) {
        alert("Error: Query Required!")
        return;
    }

    loader.style.display = 'block';
    submitButton.disabled = true;
    searchBar.disabled = true;
    promptButton.disabled = true;

	const url = test ? `http://${server_addr}:${server_port}`: 
        'https://notebooksa.jarvislabs.ai/UbU2hKCvajjAcoK_S70CPzmBgYGa-KY_xQKWmujmaZe3p5b9kpDYEbCE0vhgNdeU'
    
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

        const div = document.createElement("div");
        div.classList.add("box");

        const image = document.createElement('img')
        image.src = imageObjectURL

        const para = document.createElement("p");
        const node = document.createTextNode(`Sentence ${(parseInt(idx, 10)) + 1}: ${inputValue}`);

        para.appendChild(node);

        div.append(para)
        div.append(image);

        results.append(div);
        responseText = 'Success!!!';
    } else {
        responseText = `HTTP error: ${res.status}`;
    }

    loader.style.display = 'none';
    submitButton.disabled = false;
    searchBar.disabled = false;
    promptButton.disabled = false;

    console.log(responseText);
}