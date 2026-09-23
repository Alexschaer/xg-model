const SVG_NS = "http://www.w3.org/2000/svg";
const LEFT_POST = { cx: 36, cy: 0 };
const RIGHT_POST = { cx: 44, cy: 0 };

const state = {
  shooter: { x: 108, y: 40 },
  defenders: [],
  goalkeeper: { x: 119, y: 40 },
};

const pitch = document.querySelector("#pitch");
const players = document.querySelector("#players");
const cone = document.querySelector("#cone");
const form = document.querySelector("#options");
const xgValue = document.querySelector("#xg-value");
const xgHint = document.querySelector("#xg-hint");

let latestRequest = 0;

// StatsBomb coordinates attack towards x = 120; on screen the goal is at the top.
function toSvg({ x, y }) {
  return { cx: y, cy: 120 - x };
}

function fromSvg(svgX, svgY) {
  return { x: clamp(120 - svgY, 60, 120), y: clamp(svgX, 0, 80) };
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function clickedPosition(event) {
  const point = pitch.createSVGPoint();
  point.x = event.clientX;
  point.y = event.clientY;
  const inPitch = point.matrixTransform(pitch.getScreenCTM().inverse());
  return fromSvg(inPitch.x, inPitch.y);
}

function drawPlayer(position, className, onClick) {
  const { cx, cy } = toSvg(position);
  const circle = document.createElementNS(SVG_NS, "circle");
  circle.setAttribute("cx", cx);
  circle.setAttribute("cy", cy);
  circle.setAttribute("r", 1.3);
  circle.setAttribute("class", className);
  if (onClick) {
    circle.addEventListener("click", (event) => {
      event.stopPropagation();
      onClick();
    });
  }
  players.append(circle);
}

function render() {
  players.replaceChildren();

  const shooter = toSvg(state.shooter);
  cone.setAttribute(
    "points",
    `${shooter.cx},${shooter.cy} ${LEFT_POST.cx},${LEFT_POST.cy} ${RIGHT_POST.cx},${RIGHT_POST.cy}`,
  );

  state.defenders.forEach((defender, index) => {
    drawPlayer(defender, "defender", () => {
      state.defenders.splice(index, 1);
      refresh();
    });
  });
  drawPlayer(state.goalkeeper, "goalkeeper");
  drawPlayer(state.shooter, "shooter");
}

function shotRequest() {
  const options = form.elements;
  return {
    position: state.shooter,
    defenders: state.defenders,
    goalkeeper: state.goalkeeper,
    body_part: options.body_part.value,
    technique: options.technique.value,
    assist: options.assist.value,
    through_ball: options.through_ball.checked,
    cross: options.cross.checked,
    cut_back: options.cut_back.checked,
    first_time: options.first_time.checked,
    under_pressure: options.under_pressure.checked,
  };
}

async function updateXg() {
  const requestId = ++latestRequest;
  try {
    const response = await fetch("/api/xg", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(shotRequest()),
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const { xg } = await response.json();
    if (requestId === latestRequest) {
      showXg(xg);
    }
  } catch (error) {
    if (requestId === latestRequest) {
      xgValue.textContent = "–";
      xgHint.textContent = "The model could not be reached.";
    }
  }
}

function showXg(xg) {
  const hue = Math.round(Math.min(xg / 0.5, 1) * 120);
  xgValue.textContent = `${(xg * 100).toFixed(1)} %`;
  xgValue.style.color = `hsl(${hue}, 65%, 42%)`;
  xgHint.textContent = `About 1 goal in ${Math.max(1, Math.round(1 / xg))} such shots`;
}

function refresh() {
  render();
  updateXg();
}

pitch.addEventListener("click", (event) => {
  const position = clickedPosition(event);
  const mode = form.elements.mode.value;
  if (mode === "shooter") {
    state.shooter = position;
  } else if (mode === "defender") {
    state.defenders.push(position);
  } else {
    state.goalkeeper = position;
  }
  refresh();
});

form.addEventListener("change", refresh);

refresh();