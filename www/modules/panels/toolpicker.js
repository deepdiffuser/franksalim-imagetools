export class ToolPicker extends HTMLElement {
  constructor() {
    super();
    let shadow = this.attachShadow({mode: 'open'});
    /*
      icons from https://fonts.google.com/icons
        https://github.com/google/material-design-icons
    */
    shadow.innerHTML = `
      <style>
        :host {
          flex-basis: 60px;
          flex-grow: 0;
          flex-shrink: 0;
        }
        button {
          height: 60px;
          width: 60px;
          border: 0px;
          margin: 0px ;
          background: none;
          position: relative;
          z-index: 9;
        }
        button img {
          height: 40px;
          width: 40px;
          opacity: .5;
        }
        button[selected] {
          background-color: #ccc;
          box-shadow: -6px 8px 8px rgba(0, 0, 0, .25);
        }
        </style>

        <button id=img2imgButton title="text to image">
          <img src=/assets/imagesmode_FILL0_wght400_GRAD0_opsz48.svg>
        </button>
        <button selected id=txt2imgButton title="image to image">
          <img src=/assets/edit_document_FILL0_wght400_GRAD0_opsz48.svg>
        </button>
        <button id=inpaintingButton title="inpainting">
          <img src=/assets/brush_FILL0_wght400_GRAD0_opsz48.svg>
        </button>
        <button id=drawingButton title="draw">
          <img src=/assets/draw_FILL0_wght400_GRAD0_opsz48.svg>
        </button>
    `;
    this.shadow = shadow;
    let buttons = shadow.querySelectorAll("button");
    for (const button of buttons) {
      button.addEventListener("click", e => {
        [...buttons].map(b => { b.removeAttribute("selected") });
        button.setAttribute("selected", true);
      })
    }

    const txt2imgTool = document.getElementById("txt2img");
    const img2imgTool = document.getElementById("img2img");
    const drawingTool = document.getElementById("drawing");

    // default visibilities
    img2imgTool.style.display = "none";
    txt2imgTool.style.display = "block";
    drawingTool.style.display = "none";

    shadow.getElementById("img2imgButton").addEventListener("click", e => {
      img2imgTool.style.display = "block";
      txt2imgTool.style.display = "none";
      drawingTool.style.display = "none";
    });

    shadow.getElementById("txt2imgButton").addEventListener("click", e => {
      img2imgTool.style.display = "none";
      txt2imgTool.style.display = "block";
      drawingTool.style.display = "none";
    });

    shadow.getElementById("drawingButton").addEventListener("click", e => {
      img2imgTool.style.display = "none";
      txt2imgTool.style.display = "none";
      drawingTool.style.display = "block";
    });

    shadow.getElementById("img2imgButton").addEventListener("dragover", e => {
      shadow.getElementById("img2imgButton").dispatchEvent(new Event("click"));
    });

  }
}

window.customElements.define('fs-toolpicker', ToolPicker);
