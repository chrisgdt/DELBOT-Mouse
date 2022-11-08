class DrawElement {
  constructor(showButtonId, headerText, canvasId, nameId) {
    this.coord = { x: 0, y: 0 };
    this.lastScrollOrigin = 0;
    this.lastScroll = 0;
    this.offsetTop = 0;

    this.touchFunction = this.drawTouch.bind(this);
    this.MouseFunction = this.drawMouse.bind(this);

    this.records = new delbot.Recorder(window.screen.width, window.screen.height);

    this.createDraw(showButtonId, headerText, canvasId, nameId);
  }

  createDraw(showButtonId, headerText, canvasId, nameId, buttons='') {
    // TODO: if canvas open, forbid background operations (remove html in local then navigate)
    this.drawElement = document.createElement('div');
    this.drawElement.id = nameId;
    this.drawElement.innerHTML = `<div style="text-align: center"><table cellpadding="0" cellspacing="0" style="margin-bottom:20px;position:relative;margin-left:auto;margin-right:auto;"><tr><td style="background-color: powderblue">${headerText}</td></tr><tr><td style="padding-left:10px;padding-right:10px;background-color:#999999;"><canvas id="${canvasId}"></canvas><div>${buttons}</div></td></tr><tr><td style="background-color: powderblue"><p></p></td></tr></table></div>`;
    this.drawElement.firstElementChild.firstElementChild.style.marginTop = "450px";
    // usually, document.body.scrollHeight is good except if the body is not high enough (no scroll bar)
    let height = Math.max(document.body.scrollHeight,
      document.body.offsetHeight,
      document.documentElement.clientHeight,
      document.documentElement.scrollHeight,
      document.documentElement.offsetHeight);
    this.drawElement.style.height = height+'px';
    this.drawElement.style.width  = "100%";
    this.drawElement.style.zIndex = "1000";
    this.drawElement.style.position = "absolute";
    this.drawElement.style.backgroundColor = 'rgba(0,0,0,0.8)';
    this.drawElement.style.display = 'none';

    document.body.prepend(this.drawElement);
    document.getElementById(showButtonId).addEventListener("click", this.showDraw.bind(this));

    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext("2d");

    // Track when drawing only
    this.canvas.addEventListener("mousedown", e => {this.start(e.clientX, e.clientY, e.timeStamp, true);});
    this.canvas.addEventListener("touchstart", e => {this.start(e.touches[0].clientX, e.touches[0].clientY, e.timeStamp, false);});
    this.canvas.addEventListener("resize", this.resize.bind(this));

    document.addEventListener("mouseup", e => {this.stop(e.clientX, e.clientY, e.timeStamp, true);});
    document.addEventListener("touchend", e => {this.stop(Number.NaN, Number.NaN, e.timeStamp, false);});

    this.resize();
  }

  isShown() {
    return this.drawElement.style.display !== 'none';
  }

  showDraw() {
    this.drawElement.style.display = '';

    const position = window.scrollY;
    this.drawElement.firstElementChild.firstElementChild.style.marginTop = 50 + position + "px";
    this.lastScrollOrigin = position;
    this.lastScroll = position;
    this.offsetTop = this.drawElement.firstElementChild.getBoundingClientRect().top;
    document.addEventListener("scroll", e => {this.lastScroll = window.scrollY;});
  }

  hideDraw() {
    this.drawElement.style.display = 'none';

    document.removeEventListener("scroll", e => {this.lastScroll = window.scrollY;});
    this.clearDraw();
  }

  clearDraw() {
    // clear canvas on close
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  resize() {
    this.canvas.width = window.innerWidth/1.7;
    this.canvas.height = window.innerHeight/1.7;
  }

  reposition(x, y) {
    this.coord.x = Math.round(x - this.canvas.offsetLeft - this.drawElement.firstElementChild.firstElementChild.offsetLeft);
    this.coord.y = Math.round(y - this.canvas.offsetTop - (this.lastScrollOrigin - this.lastScroll) - this.offsetTop);
  }

  start(x, y, timeStamp, isMouse=true) {
    if (this.isShown()) {
      this.records.addRecord({
        time: timeStamp,
        type: "Pressed"+(isMouse ? "" : "Touch"),
        x, y
      });
      this.canvas.addEventListener("touchmove", this.touchFunction);
      this.canvas.addEventListener("mousemove", this.MouseFunction);
      this.reposition(x, y);
    }
  }

  stop(x, y, timeStamp, isMouse=true) {
    if (this.isShown()) {
      this.records.addRecord({
        time: timeStamp,
        type: "Released"+(isMouse ? "" : "Touch"),
        x, y
      });
      this.canvas.removeEventListener("touchmove", this.touchFunction);
      this.canvas.removeEventListener("mousemove", this.MouseFunction);
    }
  }

  drawTouch(event) {
    event.preventDefault();
    //this.records.push(`${event.timeStamp},${event.timeStamp},NoButton,MoveTouch,${event.touches[0].clientX},${event.touches[0].clientY}`);
    this.draw(event.touches[0].clientX, event.touches[0].clientY, event.timeStamp,"MoveTouch");
  }

  drawMouse(event) {
    //this.records.push(`${event.timeStamp},${event.timeStamp},NoButton,Move,${event.clientX},${event.clientY}`);
    this.draw(event.clientX, event.clientY, event.timeStamp, "Move");
  }

  draw(x, y, time, type) {
    this.ctx.beginPath();
    this.ctx.lineWidth = 4;
    this.ctx.lineCap = "round";
    this.ctx.strokeStyle = "#ACD3ED";
    this.ctx.moveTo(this.coord.x, this.coord.y);
    this.reposition(x, y);
    this.ctx.lineTo(this.coord.x, this.coord.y);
    this.ctx.stroke();
    this.records.addRecord({time, type, x, y});
  }
}

export class DrawElementSample extends DrawElement {
  constructor(showButtonId, headerText, canvasId, nameId) {
    super(showButtonId, headerText, canvasId, nameId);
    this.nameId = nameId;
  }

  createDraw(showButtonId, headerText, canvasId, nameId, buttons="") {
    super.createDraw(showButtonId, headerText, canvasId, nameId,
      buttons=`<input id="hideDraw_${canvasId}" type="submit" value="Fermer"><input id="exportDraw_${canvasId}" type="submit" value="Exporter"><input id="clearDraw_${canvasId}" type="submit" value="Effacer">`);

    document.getElementById("hideDraw_"+canvasId).addEventListener("click", e => {this.hideDraw();this.saveRecords();});
    document.getElementById("exportDraw_"+canvasId).addEventListener("click", e => {this.clearDraw();this.saveRecords();});
    document.getElementById("clearDraw_"+canvasId).addEventListener("click", e => {this.clearDraw();this.records.clearRecord();});
  }

  saveRecords() {
    if (this.records.getRecords().length > 10) {
      const element = document.createElement('a');
      const recordsString = [`resolution:${this.records.normalizer[0]},${this.records.normalizer[1]}`];
      for (let line of this.records.getRecords()) {
        recordsString.push(`${line.time},${line.type},${line.x},${line.y}`)
      }
      element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(recordsString.join("\n")));
      element.setAttribute('download', `${this.nameId}_${Date.now()}.txt`);

      element.style.display = 'none';
      document.body.appendChild(element);
      element.click();
      document.body.removeChild(element);
    }
    this.records.clearRecord();
  }
}

export class DrawElementTesting extends DrawElement {
  constructor(showButtonId, headerText, canvasId, nameId, model) {
    super(showButtonId, headerText, canvasId, nameId);
    this.model = model;
  }

  createDraw(showButtonId, headerText, canvasId, nameId, buttons="") {
    super.createDraw(showButtonId, headerText, canvasId, nameId,
      buttons=`<input id="validateDraw_${canvasId}" type="submit" value="Valider">`);

    document.getElementById("validateDraw_"+canvasId).addEventListener("click", e => {
      this.validate().then((v => {
        this.clearDraw();
        this.records.clearRecord();

        if (v.reason === delbot.Recorder.notEnoughProvidedDatas) {
          alert("DrawCaptcha échoué : Merci de dessiner davantage.");
        } else if (!v.result) {
          alert("DrawCaptcha échoué : Trop ressemblant à un bot.");
        } else {
          this.hideDraw();
        }
      }));
    });
  }

  async validate() {
    return this.records.isHuman(this.model, .3);
  }
}
