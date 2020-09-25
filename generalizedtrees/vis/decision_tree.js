// Global settings
const width = 954;
const node_width = 100;
const node_height = 100;

// Functions

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
    
    fillNodes(node);

    return svg.node();
}

// Function that dispatches different node filling calls
function fillNodes(nodeSelection){

    fillSplitNodes(nodeSelection.filter(d => !!d.children))
    
    fillPNodes(nodeSelection.filter(d => !d.children && d.data.probabilities))

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
function fillPNodes(nodeSelection){
    // TODO: pie chart
    nodeSelection.append("circle")
        .attr("fill", "#999")
        .attr("r", 2.5)
    nodeSelection.append("text")
        .attr("dy", "0.31em")
        .attr("y", 6)
        .attr("text-anchor", "middle")
        .text(d => JSON.stringify(d.data.probabilities))
        .clone(true).lower()
        .attr("stroke", "white");
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
    
    bars.append("rect")
        .attr("fill", d => d.value < 0 ? "#900" : "#090")
        .attr("y", -r_height/2)
        .attr("height", r_height)
        .attr("x", d => d.value < 0 ? d.value * bars_xscale : 0)
        .attr("width", d => Math.abs(d.value) * bars_xscale)
    
    bars.append("text")
        .attr("dy", "0.31em")
        .attr("dx", d => d.value < 0 ? "0.31em" : "-0.31em")
        .attr("text-anchor", d => d.value < 0 ? "start" : "end")
        .text(d => d.label)
        .clone(true).lower()
        .attr("stroke", "white");
}