var isNodeJs = (typeof window) === 'undefined'? true : false;

if (isNodeJs) {
  var Benchmark = require('benchmark');
  var ncvslideio = require('../../opencv');
} else {
  var paramsElement = document.getElementById('params');
  var runButton = document.getElementById('runButton');
  var logElement = document.getElementById('log');
}

function perf() {

  console.log('opencv.js loaded');
  if (isNodeJs) {
    global.ncvslideio = ncvslideio;
  } else {
    runButton.removeAttribute('disabled');
    runButton.setAttribute('class', 'btn btn-primary');
    runButton.innerHTML = 'Run';
  }
  let totalCaseNum, currentCaseId;


  function addCountNonZeroCase(suite) {
    suite.add('countNonZero', function() {
      ncvslideio.countNonZero(mat);
    }, {
      'setup': function() {
        let size = this.params.size;
        let mat = ncvslideio.Mat.eye(size[0], size[1], ncvslideio.CV_64F);
      }, 'teardown': function() {
        mat.delete();
      }
    });
  }

  function addMatDotCase(suite) {
    suite.add('Mat::dot', function() {
      mat.dot(matT);
    }, {
      'setup': function() {
        let size = this.params.size;
        let mat = ncvslideio.Mat.ones(size[0], size[1], ncvslideio.CV_64FC1);
        let matT = mat.t();
      }, 'teardown': function() {
        mat.delete();
        matT.delete();
      }
    });
  }

  function addSplitCase(suite) {
    suite.add('Split', function() {
      ncvslideio.split(mat, planes);
    }, {
      'setup': function() {
        let size = this.params.size;
        let mat = ncvslideio.Mat.ones(size[0], size[1], ncvslideio.CV_64FC3);
        let planes = new ncvslideio.MatVector();
      }, 'teardown': function() {
        mat.delete();
        planes.delete();
      }
    });
  }

  function addMergeCase(suite) {
    suite.add('Merge', function() {
      ncvslideio.merge(planes, mat);
    }, {
      'setup': function() {
        let size = this.params.size;
        let mat = new ncvslideio.Mat();
        let mat1 = ncvslideio.Mat.ones(size[0], size[1], ncvslideio.CV_64FC3);
        let planes = new ncvslideio.MatVector();
        ncvslideio.split(mat1, planes);
      }, 'teardown': function() {
        mat.delete();
        mat1.delete();
        planes.delete();
      }
    });
  }

  function setInitParams(suite, sizeArray) {
    for( let i =0; i < suite.length; i++) {
      suite[i].params = {
        size: sizeArray
      };
    }
  }

  function log(message) {
    console.log(message);
    if (!isNodeJs) {
      logElement.innerHTML += `\n${'\t' + message}`;
    }
  }

  function setBenchmarkSuite(suite) {
    suite
    // add listeners
    .on('cycle', function(event) {
      ++currentCaseId;
      let size = event.target.params.size;
      log(`=== ${event.target.name} ${currentCaseId} ===`);
      log(`params: (${parseInt(size[0])}x${parseInt(size[1])})`);
      log('elapsed time:' +String(event.target.times.elapsed*1000)+' ms');
      log('mean time:' +String(event.target.stats.mean*1000)+' ms');
      log('stddev time:' +String(event.target.stats.deviation*1000)+' ms');
      log(String(event.target));
    })
    .on('error', function(event) { log(`test case ${event.target.name} failed`); })
    .on('complete', function(event) {
      log(`\n ###################################`)
      log(`Finished testing ${event.currentTarget.length} cases \n`);
      if (!isNodeJs) {
        runButton.removeAttribute('disabled');
        runButton.setAttribute('class', 'btn btn-primary');
        runButton.innerHTML = 'Run';
      }
    });
  }

  function genBenchmarkCase(paramsContent) {
    let suite = new Benchmark.Suite;
    var sizeArray;
    totalCaseNum = 4;
    currentCaseId = 0;
    if (/\([0-9]+x[0-9]+\)/g.test(paramsContent.toString())) {
      let params = paramsContent.toString().match(/\([0-9]+x[0-9]+\)/g)[0];
      let sizeStrs = (params.match(/[0-9]+/g) || []).slice(0, 2).toString().split(",");
      sizeArray = sizeStrs.map(Number);
    } else {
      log("no getting invalid params, run all the cases with Mat of shape (1000 x 1000)");
      sizeArray = [1000, 1000];
    }
    addCountNonZeroCase(suite);
    addMatDotCase(suite);
    addSplitCase(suite);
    addMergeCase(suite);
    setInitParams(suite, sizeArray)
    setBenchmarkSuite(suite);
    log(`Running ${totalCaseNum} tests from 64-bit intrinsics`);
    suite.run({ 'async': true }); // run the benchmark
  }


  // set test filter params
  if (isNodeJs) {
    const args = process.argv.slice(2);
    let paramsContent = '';
    if (/--test_param_filter=\([0-9]+x[0-9]+,[\ ]*\w+\)/g.test(args.toString())) {
      paramsContent = args.toString().match(/\([0-9]+x[0-9]+,[\ ]*\w+\)/g)[0];
    }
    genBenchmarkCase(paramsContent);
  } else {
    runButton.onclick = function()　{
      let paramsContent = paramsElement.value;
      genBenchmarkCase(paramsContent);
      if (totalCaseNum !== 0) {
        runButton.setAttribute("disabled", "disabled");
        runButton.setAttribute('class', 'btn btn-primary disabled');
        runButton.innerHTML = "Running";
      }
    }
  }
};

async function main() {
  if (ncvslideio instanceof Promise) {
    ncvslideio = await ncvslideio;
    perf();
  } else {
    ncvslideio.onRuntimeInitialized = perf;
  }
}

main();