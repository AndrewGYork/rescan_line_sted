// All figures interactively loads static images.
// Check if our big images are available locally or remotely:
var big_image_directory = "./../big_images";
var img = new Image();
img.onerror = function() {
  window.big_image_directory = "https://andrewgyork.github.io/line-rescan-sted-data/big_images";
  img.src = big_image_directory + "/figure_1/point_1p00exc_9p00dep_008samps_001pulses.svg"
}
img.onload = function() {
  console.log("Loading interactive images from: " + big_image_directory)
}
img.src = big_image_directory + "/figure_1/point_1p00exc_9p00dep_008samps_001pulses.svg"

// Now that we know where our images are, we can load them:
function update_figure_1() {
  var exc = document.getElementById("Figure 1 excitation").value;
  var dep = document.getElementById("Figure 1 depletion").value;
  var smp = document.getElementById("Figure 1 samples").value;
  var pul = document.getElementById("Figure 1 pulses").value;
  var shp = document.getElementById("Figure 1 shape").value;
  var filename = big_image_directory + "/figure_1/" + shp + exc + dep + smp + pul + ".svg";
  var image = document.getElementById("Figure 1 image");
  image.src = filename;
}

function update_figure_2() {
  var res = document.getElementById("Figure 2 resolution").value;
  var img = document.getElementById("Figure 2 image").value;
  var scn = document.getElementById("Figure 2 scan type").value;
  var filename = big_image_directory + "/figure_2/figure_2_" + img + res + scn;
  var gif = document.getElementById("Figure 2 gif");
  var svg = document.getElementById("Figure 2 svg");
  gif.src = filename.concat(".gif");
  svg.src = filename.concat(".svg");
}

function update_figure_3() {
  var mth = document.getElementById("Figure 3 imaging method").value;
  var res = document.getElementById("Figure 3 resolution").value;
  var img = document.getElementById("Figure 3 test image").value;
  var fov = document.getElementById("Figure 3 field of view").value;
  var num = document.getElementById("Figure 3 number of angles").value;
  var filename = big_image_directory + "/figure_3/" + mth;
  if ((mth === 'descan_line_') || (mth === 'rescan_line_')) {
    filename = filename.concat(num);
  }
  filename = filename.concat(img + fov + res + ".mp4");
  console.log(filename);
  var vid = document.getElementById("Figure 3 vid");
  vid.src = filename;
}
