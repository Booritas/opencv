if (typeof window === 'undefined') {
  var ncvslideio = require("../opencv");
  if (ncvslideio instanceof Promise) {
    loadOpenCV();
  } else {
    ncvslideio.onRuntimeInitialized = perf;
  }
}

let gCvSize;

function getCvSize() {
  if (gCvSize === undefined) {
    gCvSize = {
      szODD: new ncvslideio.Size(127, 61),
      szQVGA: new ncvslideio.Size(320, 240),
      szVGA: new ncvslideio.Size(640, 480),
      szSVGA: new ncvslideio.Size(800, 600),
      szqHD: new ncvslideio.Size(960, 540),
      szXGA: new ncvslideio.Size(1024, 768),
      sz720p: new ncvslideio.Size(1280, 720),
      szSXGA: new ncvslideio.Size(1280, 1024),
      sz1080p: new ncvslideio.Size(1920, 1080),
      sz130x60: new ncvslideio.Size(130, 60),
      sz213x120: new ncvslideio.Size(120 * 1280 / 720, 120),
    };
  }

  return gCvSize;
}

async function loadOpenCV() {
  ncvslideio = await ncvslideio;
}

if (typeof window === 'undefined') {
  exports.getCvSize = getCvSize;
}