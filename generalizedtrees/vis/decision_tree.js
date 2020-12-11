/* Template for visualizing generalized decitsion trees

Licensed under the BSD 3-Clause License
Copyright (c) 2020, Yuriy Sverchkov
*/

// Global settings
const width = 954;
const node_width = 100;
const node_height = 100;

// Functions

// Tooltip event handlers
function t_mouseover(tooltip){
    return function (e, d) {

        // Clear prior contents
        tooltip.selectAll("*").remove();

        // Ugly sample listing
        if (d.data.training_samples) {
            const samples_table = tooltip.append("table");
            const samples_header = samples_table.append("thead")
                .style("font-weight", "bold")
                .append("tr");
            samples_header.append("td").text("counts");
            samples_header.selectAll("td.counts")
                .data(d.data.training_samples)
                .join("td").attr("class", "counts").text(d => d.label);
            const samples_tbody = samples_table.append("tbody");
            const training_row = samples_tbody.append("tr");
            training_row.append("td").style("font-weight", "bold").text("training");
            training_row.selectAll("td.counts")
                .data(d.data.training_samples)
                .join("td").attr("class", "counts").text(d => d.count);
            const gen_row = samples_tbody.append("tr");
            gen_row.append("td").style("font-weight", "bold").text("generated");
            gen_row.selectAll("td.counts")
                .data(d.data.generated_samples)
                .join("td").attr("class", "counts").text(d => d.count)
        }
    
        
        // Fill contents depending on node type
        if (d.children){
            // fetch split feature annotation
            if (d.data.feature_annotation){
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

        tooltip.style("visibility", "visible");
        
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

    // Links are relevant only if data has children
    if (data.children) {
    
        // Draw link core
        const link = g.append("g")
            .attr("fill", "none")
            .attr("stroke", "black")
            .attr("stroke-opacity", 0.4)
            .attr("stroke-dasharray", ("5,3"))
            .attr("stroke-width", 0.5)
            .selectAll("path")
            .data(root.links())
            .join("path")
            .attr("d", d3.linkVertical()
                .x(d => d.x)
                .y(d => d.y));
        
        // Draw sample flow.
        // Total link width is limited by node width
        total_samples = 0
        // Hacky, to fix later
        targets = []
        data.children.forEach(function(child){
            if (child.training_samples) data.training_samples.forEach(function(label){
                total_samples += label.count;
                if (!targets.includes(label.label)) targets.push(label.label)
            })
            if (child.generated_samples) data.generated_samples.forEach(function(label){
                total_samples += label.count;
                if (!targets.includes(label.label)) targets.push(label.label)
            })
        })
        sample_scale = node_width / total_samples;

        // Link drawing is:
        // [dashed boundary, space
        // for k: training label k, space,
        // for k: generated label k, space
        // .. dashed boundary]
        const trace_generated = false;
        cosmetic_links = [];
        
        root.descendants().forEach(tree_node => {
            if (tree_node.children){
                let links = [];
                let source_offset = 0;
                //let source_counts = {'training': {}, 'generated': {}};
                //let children_counts = [];
                tree_node.children.forEach(child => {
                    let target_offset = 0;
                    let child_links = [];
                    if (child.data.training_samples)
                        child.data.training_samples.forEach(entry => {
                            let hw = entry.count * sample_scale / 2;
                            target_offset += hw;
                            source_offset += hw;
                            this_link = {
                                'label': entry.label,
                                'type': 'training',
                                'count': entry.count,
                                'width': entry.count * sample_scale,
                                'source': {
                                    'center_x': tree_node.x,
                                    'y': tree_node.y,
                                    'offset': source_offset
                                },
                                'target': {
                                    'center_x': child.x,
                                    'y': child.y,
                                    'offset': target_offset
                                }
                            }
                            cosmetic_links.push(this_link),
                            links.push(this_link)
                            child_links.push(this_link)
                            target_offset += hw + 1;
                            source_offset += hw + 1;
                        });
                    if (child.data.generated_samples && trace_generated)
                        child.data.generated_samples.forEach(entry => {
                            let hw = entry.count * sample_scale / 2;
                            target_offset += hw;
                            source_offset += hw;
                            this_link = {
                                'label': entry.label,
                                'type': 'generated',
                                'count': entry.count,
                                'width': entry.count * sample_scale,
                                'source': {
                                    'center_x': tree_node.x,
                                    'y': tree_node.y,
                                    'offset': source_offset
                                },
                                'target': {
                                    'center_x': child.x,
                                    'y': child.y,
                                    'offset': target_offset
                                }
                            }
                            cosmetic_links.push(this_link),
                            links.push(this_link)
                            child_links.push(this_link)
                            target_offset += hw + 1;
                            source_offset += hw + 1;
                        });
                    // Properly center links on target end
                    // Target width is target_offset
                    child_links.forEach(ln =>
                        ln.target.x = ln.target.center_x - target_offset / 2 + ln.target.offset)
                })

                // Properly center links on target end
                // Source width is source_offset
                links.forEach(ln =>
                    ln.source.x = ln.source.center_x - source_offset / 2 + ln.source.offset)
            }
        })

        // Draw bands
        g.append("g")
            .attr("fill", "none")
            .attr("stroke", "#555")
            .attr("stroke-opacity", 0.7)
            //.attr("stroke-dasharray", ("5,3"))
            .selectAll("path")
            .data(cosmetic_links)
            .join("path")
            .attr("stroke-width", d => d.width)
            .attr("d", d3.linkVertical()
                .x(d => d.x)
                .y(d => d.y));
        
        // Draw labels
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
    }

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
    
    fillPNodes(nodeSelection.filter(d => !d.children && d.data.model.estimate), legend)

    fillLRNodes(nodeSelection.filter(d => !d.children && d.data.model.coefficients))
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
        d.data.model.estimate.forEach(v => targets.add(v.label)))
    const color = d3.scaleOrdinal().domain(targets).range(d3.schemeCategory10)

    // Helpers
    const pie = d3.pie().sort(null).value(d => d.value)
    const arc = d3.arc().innerRadius(0).outerRadius(Math.min(node_height, node_width)/4)

    // Draw pie charts
    slices = nodeSelection.selectAll("path")
        .data(d => pie(d.data.model.estimate))
        .join("path")
        .attr("fill", d => color(d.data.label))
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
        d.data.model.coefficients.forEach(function(v){
            if (v.value < min_v) min_v = v.value;
            if (v.value > max_v) max_v = v.value;
        })
    })

    const bars_xscale = node_width / (max_v - min_v);
    const bars_xshift = 0 - (max_v + min_v) / 2;
    
    bars = nodeSelection.selectAll("g")
        .data(d => d.data.model.coefficients)
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