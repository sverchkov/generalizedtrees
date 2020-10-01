// Global settings
const width = 954;
const node_width = 100;
const node_height = 100;

// Functions

// Tooltip event handlers
function t_mouseover(tooltip){
    return function (e, d) {

        let show = false;

        // Clear prior contents
        tooltip.selectAll("*").remove();

        // Ugly sample listing
        const samples_table = tolltip.append("table");
        const samples_header = samples_table.append("thead").append("tr");
        samples_header.append("td").text("counts");
        samples_header.selectAll("td")
            .data(d.data.training_samples)
            .enter()
            .append("td").text(d => d.label);
        const samples_tbody = samples_table.append("tbody");
        const training_row = samples_tbody.append("tr");
        training_row.append("td").text("training");
        training_row.selectAll("td")
            .data(d.data.training_samples)
            .enter()
            .append("td").text(d => d.count);
        const gen_row = samples_tbody.append("tr");
        gen_row.append("td").text("generated");
        gen_row.selectAll("td")
            .data(d.data.generated_samples)
            .enter()
            .append("td").text(d => d.count)
    
        
        // Fill contents depending on node type
        if (d.children){
            // fetch split feature annotation
            if (d.data.feature_annotation){
                show = true;
                const lines = tooltip.selectAll("div")
                    .data(d.data.feature_annotation)
                    .join("div");
                lines.append("span").style("font-weight", "bold").text(d => d.annotation + ": ");
                lines.append("span").text(d => d.value);
            }
        } else {
            // TODO: contents for logistic leaves

            // Probability leaves:
            if (d.data.probabilities){
                show = true;
                const table = tooltip.append("table");
                //const header = table.append("thead").append("tr");
                //header.append("td");
                //header.append("td").text("probability");
                const rows = table.append("tbody")
                    .selectAll("tr")
                    .data(d.data.probabilities)
                    .join("tr");
                rows.append("td").style("font-weight", "bold").text(d => d.target + ":");
                rows.append("td").text(d => d.value);
            }
        }

        if (show) {
            tooltip.style("visibility", "visible");
        }
        return tooltip;
    }
}

function t_mouseout(tooltip){
    return () => tooltip.style("visibility", "hidden");
}

function t_mousemove(tooltip){
    return (e) => tooltip
        .style("left", (e.pageX + 20) + "px")
        .style("top", (e.pageY + 20) + "px");
}

// Compute the tree layout
function tree(data) {
    const root = d3.hierarchy(data);

    return d3.tree().size([width - node_width, root.height * 2 * node_height])(root);
}

// Backwards compatibility werapper
function draw_it(){
    d3.select("#drawing").append(() => draw_tree(data))
}

// Draw the tree
function draw_tree(data){
    
    const root = tree(data)

    // Figure out y limits
    let y0 = Infinity;
    let y1 = -Infinity;
    let x0 = Infinity;
    let x1 = -Infinity;
    root.each(d => {
        if (d.y > y1) y1 = d.y;
        if (d.y < y0) y0 = d.y;
        if (d.x > x1) x1 = d.x;
        if (d.x < x0) x0 = d.x;
    });

    const svg = d3.create("svg:svg")
        .attr("viewBox", [-node_width/2, -node_height/2, width, y1 - y0 + node_height]);
    
    // Create a legend in the top left
    const legend = svg.append("g")
        .attr("font-family", "sans-serif")
        .attr("font-size", 10);
    
    // Containing group for the tree
    const g = svg.append("g")
        .attr("font-family", "sans-serif")
        .attr("font-size", 10)
        .attr("transform", `translate(${0},${0})`);
        
    const link = g.append("g")
        .attr("fill", "none")
        .attr("stroke", "#555")
        .attr("stroke-opacity", 0.4)
        .attr("stroke-width", 1.5)
        .selectAll("path")
        .data(root.links())
        .join("path")
        .attr("d", d3.linkVertical()
            .x(d => d.x)
            .y(d => d.y));
    
    const linkLabel = g.append("g")
        .selectAll("text")
        .data(root.links())
        .join("text")
        .attr("dy", "0.31em")
        .attr("x", d => (d.source.x + d.target.x) / 2)
        .attr("y", d => (d.source.y + d.target.y) / 2)
        .attr("text-anchor", "middle")
        .text(d => d.target.data.branch)
        .clone(true).lower()
        .attr("stroke", "white");
    
    const node = g.append("g")
        .attr("stroke-linejoin", "round")
        .attr("stroke-width", 3)
        .selectAll("g")
        .data(root.descendants())
        .join("g")
        .attr("transform", d => `translate(${d.x},${d.y})`);

    fillNodes(node, legend);

    // Tooltip
    const tooltip = d3.select('body').append('div')
        .attr("id", "tooltip")
        .style("position", "absolute")
        .style("visibility", "hidden")
        .style("background-color", "white")
        .style("border", "solid")
        .style("border-width", "1px")
        .style("border-radius", "5px")
        .style("font-family", "sans-serif")
        .style("font-size", "10pt")
        .style("padding", "5px");

    // Tooltip events
    node.on('mouseover', t_mouseover(tooltip))
        .on('mouseout', t_mouseout(tooltip))
        .on('mousemove', t_mousemove(tooltip));
    
    return svg.node();
}

// Function that dispatches different node filling calls
function fillNodes(nodeSelection, legend){

    fillSplitNodes(nodeSelection.filter(d => !!d.children))
    
    fillPNodes(nodeSelection.filter(d => !d.children && d.data.probabilities), legend)

    fillLRNodes(nodeSelection.filter(d => !d.children && d.data.logistic_model))
}

// Draws split nodes
function fillSplitNodes(nodeSelection){
    nodeSelection.append("circle")
        .attr("fill", "#555")
        .attr("r", 2.5);
    nodeSelection.append("text")
        .attr("dy", "0.31em")
        .attr("y", -6)
        .attr("text-anchor", "middle")
        .text(d => d.data.split)
        .clone(true).lower()
        .attr("stroke", "white");
}

// Draws probability leaf nodes
function fillPNodes(nodeSelection, legend){

    // Create scale on which targets are listed
    const targets = new Set();
    nodeSelection.data().forEach(d =>
        d.data.probabilities.forEach(v => targets.add(v.target)))
    const color = d3.scaleOrdinal().domain(targets).range(d3.schemeCategory10)

    // Helpers
    const pie = d3.pie().sort(null).value(d => d.value)
    const arc = d3.arc().innerRadius(0).outerRadius(Math.min(node_height, node_width)/4)

    // Draw pie charts
    slices = nodeSelection.selectAll("path")
        .data(d => pie(d.data.probabilities))
        .join("path")
        .attr("fill", d => color(d.data.target))
        .attr("stroke-width", "0.5")
        .attr("stroke", "white")
        .attr("d", arc)

    // Add pie slices to legend:
    // Some constants
    const legend_item_size = 12
    const legend_pieslice = d3.arc()
        .startAngle(0)
        .endAngle(Math.PI * 0.3)
        .innerRadius(0)
        .outerRadius(0.8 * legend_item_size)();
    // Get current legend height
    const legend_y = legend.node().getBBox().height;
    // Create legend section
    const legend_section = legend.append("g")
        .attr("transform", `translate(${0},${legend_y})`);
    legend_section.append("text")
        .attr("font-weight", "bold")
        .attr("text-anchor", "start")
        .text("Target");
    // Create legend items
    const legend_item = legend_section
        .selectAll("g")
        .data(targets)
        .join("g")
        .attr("transform", (d, i) =>
            `translate(${legend_item_size/3}, ${(i + 1.5) * legend_item_size})`);
    legend_item.append("path")
        .attr("fill", color)
        .attr("stroke-width", "0.5")
        .attr("stroke", "white")
        .attr("d", legend_pieslice);
    legend_item.append("text")
        .attr("dy", "-0.20em")
        .attr("dx", legend_item_size)
        .attr("text-anchor", "start")
        .text(d => d);
}

// Draws logistic leaf nodes
function fillLRNodes(nodeSelection){
    const r_height = 9;

    let min_v = 0;
    let max_v = 0;

    nodeSelection.data().forEach(function(d){
        d.data.logistic_model.forEach(function(v){
            if (v.value < min_v) min_v = v.value;
            if (v.value > max_v) max_v = v.value;
        })
    })

    const bars_xscale = node_width / (max_v - min_v);
    const bars_xshift = 0 - (max_v + min_v) / 2;
    
    bars = nodeSelection.selectAll("g")
        .data(d => d.data.logistic_model)
        .join("g")
        .attr("transform", function(d,i){
            return `translate(${bars_xshift},${i * r_height})`;
        });
    
    // Drawn bars
    bars.append("rect")
        .attr("fill", d => d.value < 0 ? "#900" : "#090")
        .attr("y", -r_height/2)
        .attr("height", r_height)
        .attr("x", d => d.value < 0 ? d.value * bars_xscale : 0)
        .attr("width", d => Math.abs(d.value) * bars_xscale)
    
    // Feature names (text)
    bars.append("text")
        .attr("dy", "0.31em")
        .attr("dx", d => d.value < 0 ? "0.31em" : "-0.31em")
        .attr("text-anchor", d => d.value < 0 ? "start" : "end")
        .text(d => d.label)
        .clone(true).lower()
        .attr("stroke", "white");

    // Coefficient values (text)
    bars.append("text")
        .attr("dy", "0.31em")
        .attr("dx", d => d.value < 0 ? "-0.31em" : "0.31em")
        .attr("text-anchor", d => d.value < 0 ? "end" : "start")
        .attr("x", d => d.value * bars_xscale)
        .attr("fill", d => d.value < 0 ? "#900" : "#090")
        .text(d => d.value.toExponential(3));
}