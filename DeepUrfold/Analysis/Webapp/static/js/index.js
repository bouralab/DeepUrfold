//const svg = d3.select('svg')
var canvas = document.querySelector('canvas');


var sfams = ["1.10.10.10",
             "1.10.238.10",
             "1.10.490.10",
             "1.10.510.10",
             "1.20.1260.10",
             "2.30.30.100",
             "2.40.50.140",
             "2.60.40.10",
             "3.10.20.30",
             "3.30.230.10",
             "3.30.300.20",
             "3.30.310.60",
             "3.30.1360.40",
             "3.30.1370.10",
             "3.30.1380.10",
             "3.40.50.300",
             "3.40.50.720",
             "3.80.10.10",
             "3.90.79.10",
             "3.90.420.10"];

var sfam_names = [
    "Winged helix-like DNA-binding domain",
    "EF-hand",
    "Globins",
    "Transferase",
    "Ferritin",
    "SH3",
    "OB",
    "Immunoglobulin",
    "Beta-grasp",
    "Ribosomal Protein S5",
    "KH",
    "Sm-like (domain 2)",
    "Gyrase A",
    "K Homology domain, type 1",
    "Hedgehog domain",
    "P-loop NTPases",
    "Rossmann-like",
    "Ribonuclease Inhibitor",
    "NTPase",
    "Oxidoreductase"];

var category20 = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
        '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000',
        '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff', '#000000'];

var category24 = [
    "#FF0000", "#FF7F00", "#FFD400", "#FFFF00", "#BFFF00", "#6AFF00",
    "#00EAFF", "#0095FF", "#0040FF", "#AA00FF", "#FF00AA", "#EDB9B9",
    "#E7E9B9", "#B9EDE0", "#B9D7ED", "#DCB9ED", "#8F2323", "#8F6A23",
    "#4F8F23", "#23628F", "#6B238F", "#000000", "#737373", "#CCCCCC"];


var notes = {
  'ss':"We use a the secondary structure score used throughout this paper: (# beta atoms - #alpha atoms)/(2*(# beta atoms + # alpha atomst))+0.5",
  'charge': "Each atom it annotated with the boolean feature is_positive. We sum up all of the postive atoms and take a fraction of the total number of atoms.",
  'electrostatics': "Each atom it annotated with the boolean is_electronegtive. We sum up all of the electronegtive atoms and take a fraction of the total number of atoms.",
  'sfam': "Annotated CATH Superfamily (H level)",
  'go_mf': "We use GOATOOLS to calculate enrichment for each GO term from all domains in the predicted SBM community (Leaf grouping only) using GO Slim terms from AGR. If the domain has a GO term that is enriched in its community (p_fdr_bh <= 0.05), then it is colored for the associated term. If there are multiple enriched terms, only the first is used.",
  'go_bp': "We use GOATOOLS to calculate enrichment for each GO term from all domains in the predicted SBM community (Leaf grouping only) using GO Slim terms from AGR. If the domain has a GO term that is enriched in its community (p_fdr_bh <= 0.05), then it is colored for the associated term. If there are multiple enriched terms, only the first is used.",
  'go_cc': "We use GOATOOLS to calculate enrichment for each GO term from all domains in the predicted SBM community (Leaf grouping only) using GO Slim terms from AGR. If the domain has a GO term that is enriched in its community (p_fdr_bh <= 0.05), then it is colored for the associated term. If there are multiple enriched terms, only the first is used."

}


 var sfamColors = d3.scaleOrdinal()
   .domain(sfams)
   .range(category20);

var dNodes;
var cNodes;

var mf_codes = [];
var mf_names = [];
var mf_colors;
d3.csv("/cath_mf").then(data => {
   for (var i = 0; i < data.length; i++) {
       mf_codes.push(data[i].code);
       mf_names.push(data[i].name);
   }
   d3.csv("/mf").then(data2 => {
    for (var i = 0; i < data2.length; i++) {
      if(mf_codes.indexOf(data2[i].code)<0){
        console.log("MF names", mf_names);
        mf_codes.push(data2[i].code);
        mf_names.push(data2[i].name);
      }
    }
    // mf_codes = mf_codes.filter(function(elem, pos) {
    //   return mf_codes.indexOf(elem) == pos;
    // });

    // var mf_names = mf_names.filter(function(elem, pos) {
    //   return mf_names.indexOf(elem) == pos;
    // });

    console.log("MF codes", mf_codes);
    console.log("MF names", mf_names);

    if(mf_codes.length<24){
      mf_names = mf_names.concat(["Unknown"]);
      mf_codes = mf_codes.concat(["UNK"]);
      mf_colors = d3.scaleOrdinal()
        .domain(mf_codes)
        .range(category24.slice(0, mf_codes.length-1).concat(['#989898']));
    }
    else{
      mf_names = mf_names.concat(["Unknown"]);
      mf_codes = mf_codes.concat(["UNK"]);
      mf_colors = d3.scaleOrdinal()
        .domain(mf_codes)
        .range(category24.concat(['#989898']));
    }
  })
});


var bp_codes = [];
var bp_names = [];
var bp_colors;
d3.csv("/cath_bp").then(data => {
   for (var i = 0; i < data.length; i++) {
       bp_codes.push(data[i].code);
       bp_names.push(data[i].name);
   }
   d3.csv("/bp").then(data2 => {
      for (var i = 0; i < data2.length; i++) {
        if(bp_codes.indexOf(data2[i].code)<0){
          bp_codes.push(data2[i].code);
          bp_names.push(data2[i].name);
        }
      }
      if(bp_codes.length<24){
        bp_names = bp_names.concat(["Unknown"]);
        bp_codes = bp_codes.concat(["UNK"]);
        bp_colors = d3.scaleOrdinal()
          .domain(bp_codes)
          .range(category24.slice(0, bp_codes.length-1).concat(['#989898']));
      }
      else{
        bp_names = bp_names.concat(["Unknown"]);
        bp_codes = bp_codes.concat(["UNK"]);
        bp_colors = d3.scaleOrdinal()
          .domain(bp_codes)
          .range(category24.concat(['#989898']));
      }
    })
});

var cc_codes = [];
var cc_names = [];
var cc_colors;

var web_opacity = 0.08;
var current_opacity = web_opacity;
d3.csv("/cath_cc").then(data => {
  for (var i = 0; i < data.length; i++) {
       cc_codes.push(data[i].code);
       cc_names.push(data[i].name);
   }
   d3.csv("/cc").then(data2 => {
    for (var i = 0; i < data2.length; i++) {
      if(cc_codes.indexOf(data2[i].code)<0){
        cc_codes.push(data2[i].code);
        cc_names.push(data2[i].name);
      }
    }
    if(cc_codes.length<24){
      cc_names = cc_names.concat(["Unknown"]);
      cc_codes = cc_codes.concat(["UNK"]);
      cc_colors = d3.scaleOrdinal()
        .domain(cc_codes)
        .range(category24.slice(0, cc_codes.length-1).concat(['#989898']));
    }
    else{
      cc_names = cc_names.concat(["Unknown"]);
      cc_codes = cc_codes.concat(["UNK"]);
      cc_colors = d3.scaleOrdinal()
        .domain(cc_codes)
        .range(category24.concat(['#989898']));
    }
  })
});

var current_feature = "ss";

function draw_circle_packing_chart(flare, flare_links, element, start_feature, download_opacity){
  const svg = d3.select('svg#'+element);

  if(download_opacity)
  {
    current_opacity=0.1;
  }
  console.log("using opacity", current_opacity);

  const { width } = svg.node().getBoundingClientRect()
  const { height } = svg.node().getBoundingClientRect()
  //
  // ZOOM
  //
  const zoomable_layer = svg.append('g')

  const zoom = d3
    .zoom()
    .scaleExtent([-Infinity, Infinity])
    .on('zoom', () =>
      zoomable_layer.attr('transform', d3.event.transform)
    )

  svg.call(zoom)

  const vis = zoomable_layer.append('g')
    .attr('transform', `translate(${width / 2},${height / 2})`)

  //
  // HIERARCHY
  //
  const stratify = d3
    .stratify()
    .parentId(d => d.id.substring(0, d.id.lastIndexOf('.')))

  //
  // PACK
  //
  const w = width - 8
  const h = height - 8
  const pack = d3
    .pack()
    .size([w, h])
    .padding(3)

  //
  // LINE
  //
  const line = d3
    .line()
    .curve(d3.curveBundle.beta(1))
    .x(d => d.x)
    .y(d => d.y)

  const bubble_layer = vis.append('g').attr('transform', `translate(${-w / 2},${-h / 2})`)

  d3.csv(flare).then(data =>
    d3.csv(flare_links).then(links_data => {
      //
      // LAYOUT
      //
      // circle packing
      const root = stratify(data)
        .sum(d => d.value)
        .sort((a, b) => d3.descending(a.value, b.value))

      pack(root)

      //
      // BUNDLING
      //
      // index nodes & objectify links
      const index = {}
      root.eachBefore(d => (index[d.data.id] = d))

      links_data.forEach(d => {
        d.source = index[d.source]
        d.target = index[d.target]
        return (d.path = d.source.path(d.target))
      })

      var nodes = root.descendants()

      // bubbles
      const bubbles = bubble_layer.selectAll('.bubble').data(nodes)

      const enb = bubbles
        .enter()
        .append('circle')
          .attr('class', 'bubble')
          .attr('cx', d => d.x)
          .attr('cy', d => d.y)
          .attr('r', function(d) {d.data.r=d.r; return d.r; })
          .style('fill', d => d.data.value!="" ? d3.interpolateViridis(d.data.electrostatics) : '#000000') //rgba(0,0,0,0.1)
          .style('fill-opacity', d => d.data.value!="" ? 1.0 : current_opacity)
          //.attr('fill', d => d.data.value!="" ? d3.interpolateViridis(d.data.electrostatics) : 'rgba(0,0,0,0)')
          //.attr('fill-opacity', d => d.data.value!="" ? 1.0 : 0.9)
          .on("mouseover", function(d){
            if(d.data.value==""){
              return;
            }
            d3.selectAll("path").style("stroke-width", 0);
            colorLink(d.data.id);
          })
          .on("mouseout", function(d){
            d3.selectAll("path").style("stroke-width", 1);
          })
      console.log("used opacity", current_opacity);

      // bubble_layer.selectAll('text')
    	// 	.data(nodes)
    	// 	.enter()
    	// 	.append('svg:text')
    	// 	// .attr('x', function (d) {
    	// 	// 	return d.x;
    	// 	// })
    	// 	// .attr('y', function (d) {
    	// 	// 	return d.y;
    	// 	// })
      //
      //   .attr("x", d=>d.x)
      //   .attr("dy", d=>d.y)
      //   .attr("text-anchor", "middle")
      //   //.attr("x", 0)
      //   //.attr("y", (d, i, nodes) => `${i - nodes.length / 2 + 0.8}em`)
      //    //.attr("clip-path", d => d.clipUid) // Remove this if you want labels to extend outside circles
      //    .attr("display", d => d.children && !d.children[0].children ? 'inherit' : 'none') // Uncomment this if you want to hide labels in small circles
      //    .text(d => d.id.substring(2)) // Split words into their own tspans and lines
      //    .attr("fill", "black")
      //    .attr("stroke", "#ffffff")
      //    .attr("stroke-width", 2)
      //    .attr("stroke-linecap", "round")
      //    .attr("stroke-linejoin", "round")
      //    .attr("font-size", "10")
      //    .attr("font-family", "monospace")
      //    .style("text-shadow", "-1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000");

    //      const startAngle = Math.PI*.2;
   	// const labelArc = d3.arc()
   	// 					.innerRadius(function(d) { return (d.r - 5); })
   	// 					.outerRadius(function(d) { return (d.r + 10); })
   	// 					.startAngle(startAngle)
   	// 					.endAngle(function(d) {
   	// 						const t = (d.id[0]=="0" ? d.id.substring(2) : d.id)
    //             const total = t.replace('.', '').length+0.5*(t.split('.').length - 1) ;
   	// 						const step = 10 / d.r;
   	// 						return startAngle + (total * step);
   	// 					});
    //
   	// const groupLabels = bubble_layer
   	// 	        			.selectAll(".group")
   	// 	    					.data(nodes.filter(function(d) { return d.children!==undefined && d.children && d.children.length>1; })).enter()
   	// 	    				.append("g")
   	// 	    					.attr("class", `${element}_group`)
   	// 	      					.attr("transform", function(d) { return `translate(${d.x},${d.y})`; });
   	// groupLabels
   	// 	.append("path")
   	// 		.attr("class", `${element}_group-arc`)
   	// 		.attr("id", function(d,i) { return `${element}_arc${i}`; })
   	// 		.attr("d", labelArc)
    //     .attr("display", "none");
    //
   	// groupLabels
   	// 	.append("text")
   	// 		.attr("class", `${element}_group-label`)
   	// 		.attr("x", 5)
   	// 		.attr("dy", 12)
    //  		.append("textPath")
    //  			.attr("xlink:href", function(d,i){ return `#${element}_arc${i}`;})
    //  			.text(function(d) { return d.id[0]=="0" ? d.id.substring(2) : d.id ;})
    //       .attr("stroke", "#ffffff")
    //       .attr("font-family", "monospace")
    //       .attr("font-ize", "10")
    //       .style("text-shadow", "-1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000");

          // if you don't have children, append a rect
      // node.filter(function(d){
      //   return !d.children;
      // })
      // .append("rect")
      // .attr("width", function(d) { return d.r; })
      // .attr("height", function(d) { return d.r; })
      // .attr("x", function(d) { return -d.r/2; })
      // .attr("y", function(d) { return -d.r/2; });

      enb.append('title').text(d =>
        d.id
          // .substring(d.id.lastIndexOf('.') + 1)
          // .split(/(?=[A-Z][^A-Z])/g)
          // .join(' ')
      )

      // links
      const links = bubble_layer.selectAll('.link').data(links_data)

      links
        .enter()
        .append('path')
          .attr('class', 'link')
          .attr('d', d => line(d.path))
          .style("stroke-width", 1)
          .style("stroke", d => d3.interpolatePuRd(1-d.size/1000));

      if(element=="deepurfold"){
        dNodes = nodes;
        var $table = $('#domains_table')
        console.log(nodes.filter(function(d) { return !d.children;}).map(function(d) { return d.data;}));
        $table.bootstrapTable('destroy').bootstrapTable({
            toolbar: "#toolbar",
            search: true,
            showColumns: true,
            dataUrl: "#domain_table",
            height: 500,
            showToggle: true,
            clickToSelect: true,
            showFooter: true,
            pagination: true,
            uniqueId: "cathDomain",
            columns: [
              {
                field: 'state',
                checkbox: true,
                align: 'center',
                valign: 'middle'
              },
              {field:"cathDomain", sortable:true, title:"CATH Domain"},
              {field:"sfam", sortable:true, title:"CATH Superfamily"},
              {field:"sfam_name", sortable:true, title:"CATH Superfamily Name"},
              {field:"value", sortable:true, title:"# Atoms"},
              {field:"ss", sortable:true, title:"Secondary Structure"},
              {field:"charge", sortable:true, title:"Charge"},
              {field:"electrostatics", sortable:true, title:"Electrostatics"},
              {field:"go_bp", sortable:true, title:"Biological Process", visible: false,},
              {field:"go_mf", sortable:true, title:"Molecular Function", visible: false,},
              {field:"go_cc", sortable:true, title:"Cellular Component", visible: false,},
              {field:"go_acc", sortable:true, title:"All GO Codes", visible: false,},
              {field:"id", sortable:true, title:"id"},
            ],
            data: nodes.filter(function(d) { return !d.children;}).map(function(d) {
              return {
                cathDomain: d.data.cathDomain,
                charge: parseFloat(d.data.charge).toFixed(4),
                electrostatics: parseFloat(d.data.electrostatics).toFixed(4),
                go_acc: d.data.go_acc.split("+"),
                go_bp: d.data.go_acc.split("+").map(x => bp_names[bp_codes.indexOf(x)] + " (" + x +")").join(' '),
                go_cc:  d.data.go_acc.split("+").map(x => cc_names[cc_codes.indexOf(x)] + " (" + x +")").join(' '),
                go_mf:  d.data.go_acc.split("+").map(x => mf_names[mf_codes.indexOf(x)] + " (" + x +")").join(' '),
                id: d.data.id,
                sfam: d.data.sfam,
                sfam_name: d.data.sfam_name,
                ss: parseFloat(d.data.ss).toFixed(4),
                value: d.data.value,
                };
              }),
            detailView: true,
            onExpandRow: (index, row) => {
              var html = [];
              $.each(row, function (key, value) {
                html.push('<p><b>' + key + ':</b> ' + value + '</p>');
              });
              return html.join('');
            },
            onCheck: (row, e) => {
              console.log("selected ", row);
              selectDomains(row.cathDomain);
            },
            onCheckSome: rows => {
              console.log("selected ", rows);
              var cathDomains = rows.map(r => r.cathDomain);
              selectDomains(cathDomains);
            },
            onUncheck: (row,e) => {
              unselectDomains(row.cathDomain);
            },
            onUncheckSome: rows => {
              var cathDomains = rows.map(r => r.cathDomain);
              unselectDomains(cathDomains);
            }
        });

      }
      else{
        cNodes = nodes;
      }

      changeColor(start_feature);

      document.querySelector('input').addEventListener("keypress", function(event) {
        if (event.key === "Enter") {
          event.preventDefault();
          search();
        }});

      document.addEventListener("DOMContentLoaded", function(event) {

        $("#domains_table").on("check.bs.table", function (row,e) {
          selectDomains(row.cathDomain);
        });
        $("#domains_table").on("check-some.bs.table", function (rows) {
          var cathDomains = rows.map(r => r.cathDomain);
          selectDomains(cathDomains);
        });
        $("#domains_table").on("uncheck.bs.table", function (row,e) {
          unselectDomains(row.cathDomain);
        });
        $("#domains_table").on("uncheck-some.bs.table", function (rows) {
          var cathDomains = rows.map(r => r.cathDomain);
          unselectDomains(cathDomains);
        });
        $(".btn-default").on("click", function() {
          console.log("HERE");
          $(".btn-default").removeClass("btn-primary");
          var source = $(this).find("input[name='source']").val();
          $(this).addClass("btn-primary");
          element.innerHTML= "";
          draw_circle_packing_chart(source, flare_links, element);
          changeColor(current_feature);
        });

        if(download_opacity){
          savesvg(false);
        }
      });

      if(download_opacity){
        savesvg(false);
      }

    })
  )
}

function addDeepUrfold(){
  if($("#deepurfold_button").hasClass("btn-secondary")){
    if($("#cath_button").hasClass("btn-secondary")){
      //Remove it
      $("#deepurfold_button").removeClass("active");
      $("#deepurfold_button").removeClass('btn-secondary');
      $("#deepurfold_button").addClass('btn-outline-secondary');
      d3.select("#deepurfold_div").select("svg").selectAll("*").remove();
    }
  }
  else{
    $("#deepurfold_button").removeClass("active");
    $("#deepurfold_button").removeClass('btn-outline-secondary');
    $("#deepurfold_button").addClass('btn-secondary');
    draw_circle_packing_chart("/flare", '/flare_links', "deepurfold", $(".btn-primary")[0].id, false);
    changeColor($(".btn-primary")[0].id)
  }
}

function addCATH(){
  if($("#cath_button").hasClass("btn-secondary")){
    if($("#deepurfold_button").hasClass("btn-secondary")){
      //Remove it
      $("#cath_button").removeClass("active");
      $("#cath_button").removeClass('btn-secondary');
      $("#cath_button").addClass('btn-outline-secondary');
      d3.select("#cath_div").select("svg").selectAll("*").remove()
    }
  }
  else{
    $("#cath_button").removeClass("active");
    $("#cath_button").removeClass('btn-outline-secondary');
    $("#cath_button").addClass('btn-secondary');
    draw_circle_packing_chart("/cath", '/flare_links', "cath", $(".btn-primary")[0].id, false);
    changeColor($(".btn-primary")[0].id)
  }
  $("#cath_button").trigger(onmouseout);
}

function changeColor(feature){
  current_feature = feature;

  _changeColor(feature, "deepurfold", true);
  _changeColor(feature, "cath", false);

  if($("#"+feature).hasClass("btn-primary")){
    return;
  }

  $('.btn-group-features').each(function(i, btn) {
    if(feature==btn.id){
      $("#"+btn.id).removeClass('btn-outline-primary');
      $("#"+btn.id).addClass('btn-primary');
    }
    else if($("#"+btn.id).hasClass("btn-primary")) {
      $("#"+btn.id).removeClass('btn-primary');
      $("#"+btn.id).addClass('btn-outline-primary ');
      //$("#"+btn.id).css('color', '#0d6efd');
    }
  })

}

function _changeColor(feature,element,make_legend){
  d3.select('svg#'+element).selectAll("circle")
    .transition()
    .duration(1000)
    .style("fill", function(d){
      var color;
      if(d.data.value != ""){
        if(feature=="ss"){
          color = d3.interpolateViridis(d.data.ss*0.5+0.5);
        }
        else if(feature=="charge"||feature=="electrostatics"){
          color = d3.interpolateRdBu(d.data[feature]);
        }
        else if(feature=="sfam"){
          color = sfamColors(d.data[feature]);
        }
        else if(feature=="go_cc"){
          if(d.data[feature]==""){
            color = cc_colors("UNK");
          }
          else{
            color = cc_colors(d.data[feature]);
          }
        }
        else if(feature=="go_mf"){
          if(d.data[feature]==""){
            color = mf_colors("UNK");
          }
          else{
            color = mf_colors(d.data[feature]);
          }
        }
        else if(feature=="go_bp"){
          if(d.data[feature]==""){
            color = bp_colors("UNK");
          }
          else{
            color = bp_colors(d.data[feature]);
          }
        }
        else{
          color = d3.interpolateViridis(d.data[feature]);
        }
      }
      else{
        color = 'rgba(0,0,0)';
      }
      d.data.color = color;
      return color;
    })
    //.style('fill-opacity', d => d.data.value!="" ? 1.0 : 0.8)

  if(feature=="ss"){
    var scale = d3.scaleSequential(d3.interpolateViridis)
      .domain([0,1]);
    var title = "Secondary Structure";
    var labels = ["0.00 (Mostly Alpha)", "0.25", "0.50 (Alpha/Beta)", "0.75", "1.00 (Mostly Beta)"];
    var cells = 5;
    var ascend = false;
  }
  else if(feature=="charge"||feature=="electrostatics"){
    var scale = d3.scaleSequential(d3.interpolateRdBu)
      .domain([1,0]);
    var cells = 5;
    var ascend = true;
    if(feature=="charge"){
      var title = "Charge";
      var labels = ["-1 e_c", "-0.5", "0", "+0.5", "+1 e_c"];
    }
    else{
      var title = "Electrostatic Potential";
      var labels = ["-1 k_{B}T/e_{c}", "-0.5", "0", "+0.5", "+1 k_{B}T/e_{c}"];
    }
  }
  else if(feature=="sfam"){
    var scale = sfamColors;
    var title = "Superfamilies";
    var labels = sfam_names;
    var cells = sfams.length;
    var ascend = false;
  }
  else if(feature=="go_mf"){
    var scale = mf_colors;
    var title = "GO: Molecular Function";
    var labels = mf_names;
    var cells = mf_names.length;
    var ascend = false;
  }
  else if(feature=="go_bp"){
    var scale = bp_colors;
    var title = "GO: Biological Process";
    var labels = bp_names;
    var cells = bp_names.length;
    var ascend = false;
  }
  else if(feature=="go_cc"){
    var scale = cc_colors;
    var title = "GO: Cellular Component";
    var labels = cc_names;
    var cells = cc_names.length;
    var ascend = false;
  }

  var svg = d3.select("svg#"+element);

  svg.select(".legendSequential").remove()

  svg.append("g")
    .attr("class", "legendSequential")
    .attr("transform", "translate(20,20)");

  if(make_legend){
    var legendSequential = d3.legendColor()
        .shapeWidth(30)
        .cells(cells)
        .scale(scale)
        .title(title)
        .labels(labels)
        .ascending(ascend)

    svg.select(".legendSequential")
      .call(legendSequential);

  // var totalHeight = 0;
  //
  // items = svg.selectAll('.legendCells')
  //   .each(function() {
  //       var current = d3.select(this);
  //       current.attr('transform', `translate(0, ${totalHeight}, 0)`);
  //       totalHeight += current.node().getBBox().height + 5;
  //   });

    svg.select(".legendTitle")
      .style("font-family", "Cardo")
      .style("font-weight", "900")
      .style("font", "24px sans-serif");


    svg.selectAll(".label")
      .style("font-family", "Cardo")
      .style("font", "18px sans-serif");
  }

  $("#note")[0].innerHTML = "Note: "+notes[feature];

  current_feature = feature;
}

function selectDomains(domains){
  if (domains===undefined||domains==null){
    return;
  }
  console.log(domains, domains instanceof Array);
  if(!Array.isArray(domains)){
    domains = new Array(domains);
  }

  d3.selectAll('.bubble')
    .filter(function(d) {if(domains.includes(d.data.cathDomain)){ return true;} else { return false}})
    .transition()
    .duration(750)
    .style('fill', "red")
    .style("stroke-width", 1)
    .style("stroke", "black")
    .attr('r', function(d) { console.log(d); return 5; });
}

function unselectDomains(domains){
  if (domains===undefined||domains==null){
    return;
  }
  console.log(domains, domains instanceof Array);
  if(!Array.isArray(domains)){
    domains = new Array(domains);
  }

  d3.selectAll('.bubble')
    .filter(function(d) {if(domains.includes(d.data.cathDomain)){return true;} else { return false}})
    .transition()
    .duration(750)
    .style('fill', d => d.data.color)
    .style("stroke-width", "0px")
    .attr('r', d => d.data.r)
    .selectAll("path")
      .style("stroke-width", 0);
}

function savesvg(do_download){
  d3.selectAll('.bubble')
    .style('fill', d => d.data.value!="" ? d.data.color : 'rgba(0,0,0,0)')
    .style('fill-opacity', d => d.data.value!="" ? 1.0 : 0.1);

  current_opacity = 0.1;

  console.log('set opacity');

  if(do_download){
    if($("#deepurfold_button").hasClass("btn-secondary")){
      download(document.getElementById("deepurfold_div").innerHTML, "DeepUrfold-"+$(".btn-primary")[0].id+'.svg', 'text/plain');
    }
    if($("#cath_button").hasClass("btn-secondary")){
      download(document.getElementById("cath_div").innerHTML, "CATH-"+$(".btn-primary")[0].id+'.svg', 'text/plain');
    }
    reset_opacity();
  }
}

function reset_opacity(){
  d3.selectAll('.bubble')
    .style('fill', d => d.data.value!="" ? d.data.color : 'rgba(0,0,0)')
    .style('fill-opacity', d => d.data.value!="" ? 1.0 : web_opacity);
  console.log('reset opacity');
  current_opacity = web_opacity;
}

// Function to download data to a file
function download(data, filename, type) {
  var file = new Blob([data], {type: type});
  if (window.navigator.msSaveOrOpenBlob) // IE10+
      window.navigator.msSaveOrOpenBlob(file, filename);
  else { // Others
      var a = document.createElement("a"),
              url = URL.createObjectURL(file);
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      setTimeout(function() {
          document.body.removeChild(a);
          window.URL.revokeObjectURL(url);
      }, 0);
  }
}

function search(){
  var val = document.querySelector('input').value;
  console.log("Search for ", val)

  d3.selectAll('.bubble')
    .filter(function(d) { if( d.data.cathDomain !== undefined && (d.data.id.includes(val)||d.data.sfam.includes(val)||d.data.sfam_name.includes(val))){ console.log(d.data); return true;} else { return false; }; })
    .transition()
    .duration(750)
    .style('fill', "red")
    .style('fill-opacity', 1)
    .style("stroke-width", 1)
    .style("stroke", "black");


  // d3.selectAll("path")
  //   .data(dNodes.filter(function(d) { if( d.data.cathDomain !== undefined && (d.data.id.includes(val)||d.data.sfam.includes(val)||d.data.sfam_name.includes(val))){ console.log(d.data); return true;} else { return false; }; }))
  //   .style("stroke-width", 1)
  //   .style('fill', "#FFFF00");

  //d3.selectAll(".bubbles").data(cNodes.filter(function(d) { return d.data.cathDomain !== undefined && (d.data.id.includes(val)||d.data.sfam.includes(val)||d.data.sfam_name.includes(val)); })).enter()
  //  .style('fill', "#FFFF00");

}

function colorLink(src){
  return;
  //iterate through all the links for src and target.
  console.log(src);
  var link = d3.selectAll("path").filter(function(d){
    if(d.source.id==src){
      console.log("true");
    }
    return (d.source.id==src);
    //return (d3.select(d).data()[0][0].id == src);
  }).style("stroke-width", 1);
  //for the filtered link make the stroke red.
  //d3.selectAll(link).transition().style("visibility", true);
}

function triggerDownload (imgURI) {
  var evt = new MouseEvent('click', {
    view: window,
    bubbles: false,
    cancelable: true
  });

  var a = document.createElement('a');
  a.setAttribute('download', 'sbm-circlepack-'+current_feature+'.png');
  a.setAttribute('href', imgURI);
  a.setAttribute('target', '_blank');

  a.dispatchEvent(evt);
}

function saveSVG() {
  var canvas = document.getElementById('canvas');
  var ctx = canvas.getContext('2d');
  var data = (new XMLSerializer()).serializeToString(document.querySelector('svg'));
  var DOMURL = window.URL || window.webkitURL || window;

  var img = new Image();
  img.width = width;
  img.height = height;
  var svgBlob = new Blob([data], {type: 'image/svg+xml;charset=utf-8'});
  var url = DOMURL.createObjectURL(svgBlob);

  img.onload = function () {
    canvas.width = img.width,
    canvas.height = img.height,
    ctx.drawImage(img, 0, 0);
    DOMURL.revokeObjectURL(url);

    var imgURI = canvas
        .toDataURL('image/png')
        .replace('image/png', 'image/octet-stream');

    triggerDownload(imgURI);
  };

  img.src = url;
};

function drawSVGToCanvas(svg) {
  const width=height=3000;//svg.node().getBoundingClientRect(), svg.node().getBoundingClientRect() //svg.getBoundingClientRect();
  console.log("WH", width, height)
  const serializer = new XMLSerializer();
  const copy = svg.cloneNode(true);
  const data = serializer.serializeToString(copy);
  const image = new Image();
  //image.width = width+500;//-100;
  image.height = height;//-100;
  const blob = new Blob([data], {
    type: 'image/svg+xml;charset=utf-8'
  });
  const url = URL.createObjectURL(blob);
  return new Promise(resolve => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = width+400;//+400;
    canvas.height = height;//+400;
    image.addEventListener('load', () => {
      ctx.drawImage(image, 0, 0, width, height);
      URL.revokeObjectURL(url);
      resolve(canvas);
    }, { once: true });
    image.src = url;
  })
}

async function convertSVGsToSingleImage(svgs, format = 'image/png') {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const drawSVGs = Array.from(svgs).map(svg => drawSVGToCanvas(svg))
  const renders = await Promise.all(drawSVGs);
  canvas.width = renders.map(render => render.width).reduce((a, b) => a + b, 0);
  canvas.height = Math.max(...renders.map(render => render.height));
  var currX = 0;
  console.log(currX);
  renders.forEach((render, index) => {ctx.drawImage(render, currX, 0, render.width, render.height); currX+=render.width;});
  console.log(currX);
  const source = canvas.toDataURL(format).replace(format, 'image/octet-stream');
  return source;
}

function getImages(){
  const preview = document.getElementById('canvas');
  const svgs = document.querySelectorAll('svg');

  convertSVGsToSingleImage(svgs).then(source => {
    const image = new Image();
    image.addEventListener('load', () => {
      preview.append(image);
    })
    image.src = source;
    triggerDownload(source);
  });
}
