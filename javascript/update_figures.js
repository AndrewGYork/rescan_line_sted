// All figures interactively loads static images.
// Check if our big images are available locally or remotely:
var big_image_directory = "./../big_images";
var img = new Image();
img.onerror = function() {
  window.big_image_directory = "https://andrewgyork.github.io/line-rescan-sted-data/big_images";
  img.src = big_image_directory + "/figure_1/point_1p00exc_9p00dep_008samps_001pulses.svg"
  img.onerror = ""
}
img.onload = function() {
  console.log("Loading interactive images from: " + big_image_directory)
}
img.src = big_image_directory + "/figure_1/point_1p00exc_9p00dep_008samps_001pulses.svg"

// Now that we know where our images are, we can load them:
function update_figure_1() {
  var exc = document.getElementById("Figure_1_excitation").value;
  var dep = document.getElementById("Figure_1_depletion").value;
  var smp = document.getElementById("Figure_1_samples").value;
  var pul = document.getElementById("Figure_1_pulses").value;
  var shp = document.getElementById("Figure_1_shape").value;
  var filename = big_image_directory + "/figure_1/" + shp + exc + dep + smp + pul + ".svg";
  var image = document.getElementById("Figure_1_image");
  image.src = filename;
}

function update_figure_2() {
  var res = document.getElementById("Figure_2_resolution").value;
  var img = document.getElementById("Figure_2_image").value;
  var scn = document.getElementById("Figure_2_scan_type").value;
  var filename = big_image_directory + "/figure_2/figure_2_" + img + res + scn;
  var gif = document.getElementById("Figure_2_gif");
  var svg = document.getElementById("Figure_2_svg");
  gif.src = filename.concat(".gif");
  svg.src = filename.concat(".svg");
}

function update_figure_3() {
  var mth = document.getElementById("Figure_3_imaging_method").value;
  var res = document.getElementById("Figure_3_resolution").value;
  var img = document.getElementById("Figure_3_test_image").value;
  var fov = document.getElementById("Figure_3_field_of_view").value;
  var num = document.getElementById("Figure_3_number_of_angles").value;
  var filename = big_image_directory + "/figure_3/" + mth;
  if ((mth === 'descan_line_') || (mth === 'rescan_line_')) {
    filename = filename.concat(num);
  }
  filename = filename.concat(img + fov + res + ".mp4");
  console.log(filename);
  var vid = document.getElementById("Figure_3_vid");
  vid.src = filename;
}
