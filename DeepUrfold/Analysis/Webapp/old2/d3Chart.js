var total_superfamilies = null;
var current_superfamilies = null;

var sfam_to_names = {
  "1.10.10.10": "Winged helix",
  "1.10.238.10":	"EF-hand",
  "1.10.490.10":	"Globins",
  "1.10.510.10":	"Transferase",
  "1.20.1260.10":	"Ferritin",
  "2.30.30.100":	"SH3",
  "2.40.50.140":	"OB",
  "2.60.40.10": "Immunoglobulin",
  "3.10.20.30": "Beta-grasp",
  "3.30.230.10":	"Ribosomal Protein S5; domain 2",
  "3.30.300.20":	"KH",
  "3.30.310.60":	"Sm-like",
  "3.30.1360.40":	"Gyrase A",
  "3.30.1370.10":	"K Homology 1",
  "3.30.1380.10":	"Hedgehog ",
  "3.40.50.300":	"P-loop NTPases",
  "3.40.50.720":	"Rossmann-like",
  "3.80.10.10": "Ribonuclease Inhibitor",
  "3.90.79.10": "NTP Pyrophosphohydrolase",
  "3.90.420.10":	"Oxidoreductase"
}

// -1- Create a tooltip div that is hidden by default:
var makeTooltip = function(container){
    var circletooltip = d3.select(".tooltipdiv")
      .append("div")
        .style("opacity", 0)
        .attr("class", "tooltip")
        .style("background-color", "black")
        .style("border-radius", "5px")
        .style("padding", "10px")
        .style("color", "white");
    return circletooltip;
}
// -2- Create 3 functions to show / update (when mouse move but stay on same circle) / hide the tooltip
var showTooltip = function(tooltip, d, i) {
  if (i.data.superfamily){
    tooltip
      .transition()
      .duration(200);
    tooltip
      .style("opacity", 1)
      .html("Domain: " + i.data.name + "<br>Superfamily: "+ i.data.superfamily + "<br>SS Score: "+ i.data.ss)
      .style("left", (d.pageX+30) + "px") //d3.pointer(this)[0]
      .style("top", (d.pageY+30) + "px");
  }
}

var showSfamTooltip = function(tooltip, d, i) {
  console.log("SFAM", i)
  console.log("SFAM", d)
  sfam = d.target.textContent.split(/(\s+)/)[0]
  var data = d3.json("http://www.cathdb.info/version/v4_3_0/api/rest/superfamily/"+sfam).then(data => {
    console.log(data);
    console.log(d.pageX, d.pageY)
    tooltip
      .transition()
      .duration(200);
    tooltip
      .style("opacity", 1)
      .html("Superfamily: " + data.data.superfamily_id + "<br>Name: "+ data.data.classification_name + "<br>Description: "+ data.data.classification_description)
      .style("left", (d.pageX+30) + "px") //d3.pointer(this)[0]
      .style("top", (d.pageY+30) + "px");
    console.log(tooltip);
  });
}

var moveTooltip = function(tooltip, d) {
  tooltip
    .style("left", (d.pageX+30) + "px")
    .style("top", (d.pageY+30) + "px")
}
var hideTooltip = function(tooltip, d) {
  tooltip
    .transition()
    .duration(200)
    .style("opacity", 0)
}

function drawChart(container, legend_container, data){
  data = d3.hierarchy(data)
    .sum(d => d.value)
    .sort((a, b) => b.value - a.value);

  var width = 300; //932;
  var height = 300; //932;

  var pack = d3.pack()
    .size([width - 2, height - 2])
    .padding(3);

  var root = pack(data);

  tooltip = makeTooltip(container);

  let focus = root;
  let view;

  total_superfamilies = count_leaves(root);

  const color = d3.scaleLinear()
    .domain([0, 5])
    .range(["hsl(152,80%,80%)", "hsl(228,30%,40%)"])
    .interpolate(d3.interpolateHcl)

  const format = d3.format(",d")

  const svg = d3.select(container).append("svg")
      .attr("viewBox", `-${width / 2} -${height / 2} ${width} ${height}`)
      .style("display", "block")
      .style("margin", "0 -14px")
      .style("background", color(0))
      .style("cursor", "pointer")
      .on("click", (event) => zoom(event, root));

  const node = svg.append("g")
    .selectAll("circle")
    .data(root.descendants().slice(1))
    .join("circle")
      .attr("fill", d => d3.interpolateViridis(d.data.ss))
      .style("stroke", d => "#000")
      .attr("stroke-width", d => "0.2px")
      //.attr("pointer-events", d => !d.children ? "none" : null)
      .attr("r", d => d.r)
      .on("mouseover", function(d, i) { d3.select(this).attr("stroke", "#fff").attr("fill", d => d3.interpolateViridis(d.data.ss)); showTooltip(tooltip, d, i);})
      .on("mouseout", function() { d3.select(this).attr("stroke", "#000").attr("fill", d => d3.interpolateViridis(d.data.ss)) })
      .on("mousemove", function(d){ moveTooltip(tooltip, d) } )
      .on("mouseleave", function(d){ hideTooltip(tooltip, d) } )
      .on("click", (event, d) => focus !== d && (zoom(event, d), event.stopPropagation()));

  // const label = svg.append("g")
  //     .style("font", "6px sans-serif")
  //     .attr("pointer-events", "none")
  //     .attr("text-anchor", "middle")
  //   .selectAll("text")
  //   .data(root.descendants())
  //   .join("text")
  //     .style("fill-opacity", d => d.parent === root ? 1 : 0)
  //     .style("display", d => d.parent === root ? "inline" : "none")
  //     .text(d => d.data.name);

  zoomTo([root.x, root.y, root.r * 2]);

  function zoomTo(v) {
    const k = width / v[2];

    view = v;

    //label.attr("transform", d => "translate(" + (d.x - v[0]) * k + "," + (d.y - v[1]) * k + ")");
    node.attr("transform", d => "translate(" + (d.x - v[0]) * k + "," + (d.y - v[1]) * k + ")");
    node.attr("r", d => d.r * k);
  }

  function zoom(event, d) {
    if(d.children){
        const focus0 = focus;

        focus = d;

        const transition = svg.transition()
            .duration(event.altKey ? 7500 : 750)
            .tween("zoom", d => {
              const i = d3.interpolateZoom(view, [focus.x, focus.y, focus.r * 2]);
              return t => zoomTo(i(t));
            });

        // label
        //   .filter(function(d) { return d.parent === focus || this.style.display === "inline"; })
        //   .transition(transition)
        //     .style("fill-opacity", d => d.parent === focus ? 1 : 0)
        //     .on("start", function(d) { if (d.parent === focus) this.style.display = "inline"; })
        //     .on("end", function(d) { if (d.parent !== focus) this.style.display = "none"; });


        drawDivergingHist(legend_container, d);
        if (d.data.name != "DeepUrfold"){
          drawShapemers(d.data.name);
        }
    }
    else{
        console.log(d);
        //d3.select(this).attr("stroke", "#fff")
        drawLRP(legend_container, d, total_superfamilies);

    }
  }

  drawDivergingHist(legend_container, root);

  return svg.node();
}

function drawLegend(legend_container, node){
    //sort bars based on value
    superfamilies = count_leaves(node);

    data = superfamilies.sort(function (a, b) {
        return d3.ascending(a.name, b.name);
    })

    total_local = d3.sum(data, d => d.value);
    all_data = data.map(d => {
        return [
          {"name":d.name, "category":"local", "value":d.value/total_local},
          {"name":d.name, "category":"global", "value":d.value/total_superfamilies.get(d.name, d.value)}
        ]}).reduce((prev, next) => prev.concat(next), [])


    d3.select(legend_container).html("");

    tooltip = makeTooltip(legend_container);

    // set the dimensions and margins of the graph
    var margin = {top: 50, right: 0, bottom: 0, left: 100},
        width = 400 - margin.left - margin.right,
        height = 500 - margin.top - margin.bottom;

    // set the ranges
    var y = d3.scaleBand()
              .range([height, 0])
              .padding(0.1);

    var x = d3.scaleLinear()
              .range([0, width]);

    // append the svg object to the body of the page
    // append a 'group' element to 'svg'
    // moves the 'group' element to the top left margin
    var svg = d3.select(legend_container).append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
      .append("g")
        .attr("transform",
              "translate(" + margin.left + "," + margin.top + ")");

    svg.append("text")
      .attr("x", (width / 2))
      .attr("y", 0 - (margin.top / 2))
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("text-decoration", "underline")
      .text(node.data.name + " (" + superfamilies.length + " superfamilies)");

      // format the data
      data.forEach(function(d) {
        d.value = +d.value/total_local;
      });

      // Scale the range of the data in the domains
      x.domain([0, d3.max(data, function(d){ return d.value; })])
      y.domain(data.map(function(d) { return d.name; }));
      //y.domain([0, d3.max(data, function(d) { return d.sales; })]);

      // append the rectangles for the bar chart
      svg.selectAll(".bar")
          .data(data)
        .enter().append("rect")
          .attr("class", "bar")
          //.attr("x", function(d) { return x(d.value); })
          .attr("width", function(d) {return x(d.value); } )
          .attr("y", function(d) { return y(d.name); })
          .attr("height", y.bandwidth());

      // add the x Axis
      svg.append("g")
          .attr("transform", "translate(0," + height + ")")
          .call(d3.axisBottom(x));

      // add the y Axis
      svg.append("g")
          .call(d3.axisLeft(y));


}

function drawDivergingHist(legend_container, node){
    //sort bars based on value
    current_superfamilies = count_leaves(node);

    d3.select(legend_container).html("");
    tooltip = makeTooltip(legend_container);

    console.log(current_superfamilies);

    data = current_superfamilies.sort(function (a, b) {
        return d3.ascending(a.name, b.name);
    })

    total_local = d3.sum(data, d => d.value);
    data = data.map(d => {
        global_value = total_superfamilies.find( ({ name }) => name === d.name)
        global_total = (global_value && global_value.value) || d.value;
        return [
          {"name":sfam_to_names[d.name], "category":"local", "value":d.value/total_local},
          {"name":sfam_to_names[d.name], "category":"global", "value":d.value/global_total}
        ]}).reduce((prev, next) => prev.concat(next), [])

    group_info = {
      format: ".0%",
      negative: "Local (in community)",
      positive: "Global (from superfamily)",
      negatives: ["local"],
      positives: ["global"]
    };

    console.log("DATA", data);
    console.log("group_info", group_info);

    signs = new Map([].concat(
      group_info.negatives.map(d => [d, -1]),
      group_info.positives.map(d => [d, +1])
    ))

    bias = d3.rollups(data, v => d3.sum(v, d => d.value * Math.min(0, signs.get(d.category))), d => d.name)
      .sort(([, a], [, b]) => d3.ascending(a, b))

    series = d3.stack()
        .keys([].concat(group_info.negatives.slice().reverse(), group_info.positives))
        .value(([, value], category) => signs.get(category) * (value.get(category) || 0))
        .offset(d3.stackOffsetDiverging)
      (d3.rollups(data, data => d3.rollup(data, ([d]) => d.value, d => d.category), d => d.name))

    console.log("series", series);

    width = 400
    margin = ({top: 40, right: 30, bottom: 0, left: 80})
    height = bias.length * 33 + margin.top + margin.bottom

    x = d3.scaleLinear()
      .domain(d3.extent(series.flat(2)))
      .rangeRound([margin.left, width - margin.right])

    y = d3.scaleBand()
      .domain(bias.map(([name]) => name))
      .rangeRound([margin.top, height - margin.bottom])
      .padding(2 / 33)

    // console.log("group_info.negatives.length + group_info.positives.length", group_info.negatives.length + group_info.positives.length, d3.schemeSpectral[group_info.negatives.length + group_info.positives.length]);
    //
    // color = d3.scaleOrdinal()
    //   .domain([].concat(group_info.negatives, group_info.positives))
    //   .range(d3.schemeSpectral[group_info.negatives.length + group_info.positives.length])

    formatValue = function(x){
        const format = d3.format(group_info.format || "");
        return format(Math.abs(x));
      }

    xAxis = g => g
      .attr("transform", `translate(0,${margin.top})`)
      .call(d3.axisTop(x)
          .ticks(width / 80)
          .tickFormat(formatValue)
          .tickSizeOuter(0))
      .call(g => g.select(".domain").remove())
      .call(g => g.append("text")
          .attr("x", x(0) + 20)
          .attr("y", -24)
          .attr("fill", "currentColor")
          .attr("text-anchor", "start")
          .text(group_info.positive))
      .call(g => g.append("text")
          .attr("x", x(0) - 20)
          .attr("y", -24)
          .attr("fill", "currentColor")
          .attr("text-anchor", "end")
          .text(group_info.negative))

    yAxis = g => g
      .call(d3.axisLeft(y).tickSizeOuter(0))
      .call(g => g.selectAll(".tick").data(bias).attr("transform", ([name, min]) => `translate(${x(min)},${y(name) + y.bandwidth() / 2})`))
      .call(g => g.select(".domain").attr("transform", `translate(${x(0)},0)`))

    const svg = d3.select(legend_container).append("svg")
      .attr("viewBox", [0, 0, width, height]);

    svg.append("g")
      .selectAll("g")
      .data(series)
      .join("g")
        .attr("fill", d => d.key=="local"?"steelblue":"gray")
      .selectAll("rect")
      .data(d => d.map(v => Object.assign(v, {key: d.key})))
      .join("rect")
        .attr("x", d => x(d[0]))
        .attr("y", ({data: [name]}) => y(name))
        .attr("width", d => x(d[1]) - x(d[0]))
        .attr("height", y.bandwidth())
      .on("mouseover", function(d, i) { d3.select(this).attr("fill", d => d3.interpolateViridis("cyan")); showSfamTooltip(tooltip, d, i);})
      .on("mouseout", function() { d3.select(this).attr("fill", d => d.key=="local"?"steelblue":"gray") })
      .on("mousemove", function(d) { moveTooltip(tooltip, d); } )
      .on("mouseleave", function(d) { hideTooltip(tooltip, d); }  )
      .append("title")
        .text(({key, data: [name, value]}) => `${name}
    ${formatValue(value.get(key))} ${key}`);

    svg.append("g")
        .call(xAxis);

    svg.append("g")
        .call(yAxis);

    return svg.node();
}

function count_leaves(node){
    if(node.children){
        var superfamilies = {}
        //go through all its children
        for(var i = 0; i<node.children.length; i++){
            //if the current child in the for loop has children of its own
            //call recurse again on it to decend the whole tree
            children = count_leaves(node.children[i]);
            for (var j = 0; j<children.length; j++){
                child = children[j];
                if (child.name in superfamilies) {
                    superfamilies[child.name] = superfamilies[child.name]+child.value;
                }
                else{
                    superfamilies[child.name] = child.value;
                }
            }
        }

        var arr = [];
        for (var i in superfamilies){
            arr.push({"name":i, "value":superfamilies[i]});
        }
        return arr;
    }
    //if not then it is a leaf so we count it
    else{
        return [{"name":node.data.superfamily, "value": 1}];
    }
}

$.xhrPool = [];

$.ajaxSetup({
    beforeSend: function (jqXHR) {
        $.xhrPool.push(jqXHR);
    },
    complete: function (jqXHR) {
        var i = $.xhrPool.indexOf(jqXHR);
        if (i > -1)
            $.xhrPool.splice(i, 1);
    }
});

$.xhrPool.abortAll = function () {
    $(this).each(function (i, jqXHR) {
        jqXHR.abort();
    });
    $.xhrPool.length = 0;
};

function drawShapemers(community_str){

    console.log(community_str);

    let re = /\((\d+)\.0, (\d+)\.0, (\d+)\.0, (\d+)\.0, (\d+)\.0\)/g
    var match = re.exec(community_str);
    console.log(match);
    if (match === null){
      return;
    }
    match.shift();
    var community = match.join("_");
    console.log(community);

    $("#shapemers").html("");

    d3.json('/shapemer_html/'+community+'/by_shapemer/sizes.json').then(data => {
      var max_s = data.max_s;
      var max_p = data.max_p;
      delete data.max_s;
      delete data.max_p;
      $("#dropdownMenuLink").html("Select Shapemer...")
      $(".dropdown-menu").html("");
      for(var x in Object.keys(data)){
          $(".dropdown-menu").append('<li><a class="dropdown-item" href="#">${x} (${data[x][0]} shapers across ${data[x][1]} proteins)</a></li>');
      }

      $('div.btn-group ul.dropdown-menu li a').click(function (e) {
          var $div = $(this).parent().parent().parent();
          var $btn = $div.find('button');
          var shapmer = $(this).text().split(" ")[0];
          $btn.html($(this).text() + ' <span class="caret"></span>');
          $div.removeClass('open');

          $("#shapemers").html("");

          d3.json('/shapemer_html/'+community+'/by_shapemer/'+shapemer+'.js'').then(shapemer_data => {
            Plotly.newPlot("#shapemers", data);
          };
          // $('<iframe>', {
          //    src: '/shapemer_html/'+community+'/by_shapemer/'+shapemer+'.html',
          //    id:  'shapmerFrame_'+max_s[1],
          //    frameborder: 0,
          //    width: "100%",
          //    height: 525,
          //    scrolling: "no",
          //    seamless:"seamless",
          //
          //    }).appendTo('#shapemers')

          e.preventDefault();

          return false;
      });

      $('<iframe>', {
         src: '/shapemer_html/'+community+'/by_shapemer/'+max_s[1]+'.html',
         id:  'shapmerFrame_'+community+'_'+max_s[1],
         frameborder: 0,
         width: "100%",
         height: 525,
         scrolling: "no",
         seamless:"seamless",
         }).appendTo('#shapemers')
       // $('<iframe>', {
       //    src: '/shapemer_html/'+community+'/by_shapemer/'+max_p[1]+'.html',
       //    id:  'shapmerFrame_'+community+'_'+max_p[1],
       //    frameborder: 0,
       //    width: 525,
       //    height: 525,
       //    scrolling: "no",
       //    seamless:"seamless",
       //    }).appendTo('#shapemers')
    });
}

var viewers = null;
function drawLRP(legend_container, domain, total_superfamilies){
    //$.xhrPool.abortAll();
    group = domain.parent.data.name;
    group = group.replace(/\s+/g, '-').replace(/\(|\)|,+/g, '');
    console.log("GROUP", group)
    // d3.select("#protein1234").html("");
    // d3.select("#protein1234").html("");
    //$("#protein")[0].setAttribute("style","display:inline-block;width:42%;height:500px;");
   //  viewers = null;
   //  viewers = $3Dmol.createViewerGrid(
   //      $("#protein1234"), //legend_container.slice(1), //id of div to create canvas in
   //      {
   //          rows: 4,
   //          cols: 5,
   //          control_all: true  //mouse controls all viewers
   //      },
   //      { backgroundColor: 'lightgrey' }
   // );

   viewers = [];

   d3.json('/static/lrp_by_domain/'+domain.data.name+'.json').then(data => {
     console.log(data);
     for(var i=0; i<total_superfamilies.length; i++){
        var sfam = total_superfamilies[i].name;
        row = Math.floor(i/4);
        col = i%5;
        d3.select("#viewer"+(i+1)).html("");
        console.log("curr sfams", sfam, current_superfamilies)
        if(sfam==domain.data.superfamily){
            var color = "red";
        }
        else{
          var color = "gray";
            for (var sfam_iter in current_superfamilies){
                if(sfam_iter.name == sfam){
                    color = "black";
                    break;
                }
        }

        }
        d3.select("#title"+(i+1)).html("<font color='"+color+"'>"+sfam_to_names[sfam]+"</font>");
        var viewer = $3Dmol.createViewer($("#viewer"+(i+1)));
        // $3Dmol.download("pdb:1MO8",viewer,{multimodel:true, frames:true},function(){
        // 	viewer.setStyle({}, {cartoon:{color:"spectrum"}});
        // 	viewer.render();
        // });

        //var viewer = viewers[row][col];
        var propMap = data[sfam];
        console.log(propMap);
        colorscheme = {
          'prop': "total_relevance",
          'gradient':"rwb", //new $3Dmol.Gradient.Sinebow(0,50),
          'min':d3.quantile(propMap, 0.5, d => d.props.total_relevance),
          'max':d3.quantile(propMap, 0.9, d => d.props.total_relevance)
        };
        console.log(colorscheme);
        viewer.addModel(data["pdb"],'pdb');
        viewer.mapAtomProperties(propMap);
        //viewer.setStyle({cartoon:{colorscheme:{prop:'b',gradient: new $3Dmol.Gradient.Sinebow(0,50)}}});
        viewer.setStyle({'cartoon':{'colorscheme':colorscheme}});
        viewer.zoomTo();
        viewer.render();
        viewers.push(viewer);
      }
   });

   // for(var i=0; i<total_superfamilies.length; i++){
   //    var sfam = total_superfamilies[i].name
   //    var url = '/static/lrp/'+domain.data.name+'.json';
   //    console.log("Loading", url, Math.floor(i/4))
   //    $.get(url, function(data) {
   //      console.log(this);
   //      console.log(this.url.split("/"))
   //      var curr_sfam = this.url.split("/")[4];
   //      console.log("SFAM IS", curr_sfam);
   //      for(var j=0; j<total_superfamilies.length; j++){
   //         if(total_superfamilies[j].name == curr_sfam){
   //           console.log("FOUND SFAM", curr_sfam);
   //           break;
   //         }
   //      }
   //      console.log(viewers, j, Math.floor(j/4), viewers[Math.floor(j/4)]);
   //
   //      var viewer = viewers[Math.floor(j/4)][j%5];
   //      viewer.addModel(data,'pdb');
   //      viewer.setStyle({cartoon:{colorscheme:{prop:'b',gradient: new $3Dmol.Gradient.Sinebow(0,50)}}});
   //      viewer.zoomTo();
   //      viewer.render( );
   //      console.log("Loaded struture", viewer, i, Math.floor(i/4), i%5)
   //    })
      // .error(function(){
      //   console.log("Failed downloading", url);
      // });
    //}
}

function weighted_random(total_superfamilies) {
    var i;

    var weights = [];

    for (i = 0; i < options.length; i++)
        weights[i] = 1/total_superfamilies[i].value + (weights[i - 1] || 0);

    var random = Math.random() * weights[weights.length - 1];

    for (i = 0; i < weights.length; i++)
        if (weights[i] > random)
            break;

    return total_superfamilies[i].name;
}

function weighted_random_sample(total_superfamilies, k){
    var sample = {};
    for (i=0; i<k; i++){
        sfam = weighted_random(total_superfamilies);
    }
    var superfamlies = []
    for (var j = 0; j<children.length; j++){
        child = children[j];
        if (child.name in superfamilies) {
            superfamilies[child.name] = superfamilies[child.name]+child.value;
        }
        else{
            superfamilies[child.name] = child.value;
        }
    }
}
