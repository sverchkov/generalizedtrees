// Global settings
const width = 954;
const node_width = 100;
const node_height = 100;

// Functions

// Compute the tree layout
tree = data => {
    const root = d3.hierarchy(data);

    return d3.tree().size([width - node_width, root.height * 2 * node_height])(root);
}

// Draw the tree
function draw_it(){
    
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

    const svg = d3.select("#drawing").append("svg")
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
    const r_height = 3;

    nodeSelection.append("g")
        .selectAll("rectangle")
        .data(d => d.data.logistic_model)
        .join("rectangle")
        .attr("fill", v => v < 0 ? "#900" : "#090")
        .attr("y", function(v,i){return i * r_height;})
        .attr("x", 0)
        .attr("width", v => v)
        .attr("height", r_height);
    
    nodeSelection.append("g")
        .selectAll("text")
        .data(function(d, i){

            return d.data.logistic_model;
        })
        .join("text")
        .attr("dy", "0.31em")
        .attr("y", 6)
        .attr("text-anchor", "middle")
        .text(d => JSON.stringify(d))
}

function buildNode(datum){
    construction = d3.create("svg:circle")
        .datum(datum)
        .attr("r", 4)
        .attr("fill", d => d.children ? "#900" : "#090");
    
    return construction.node();
}

// Node drawing:
function drawNode(node_data, i){

    const node = d3.select(this);

    node.append("circle")
        .attr("fill", d => d.children ? "#555" : "#999")
        .attr("r", 2.5);

    if(node_data.data.split){

        node.append("text")
            .attr("dy", "0.31em")
            .attr("y", -6)
            .attr("text-anchor", "middle")
            .text(d => d.data.split)
            .clone(true).lower()
            .attr("stroke", "white");
    }else if(node_data.data.branch){
        node.append("text")
            .attr("dy", "0.31em")
            .attr("y", -6)
            .attr("text-anchor", "middle")
            .text(d => d.data.branch)
            .clone(true).lower()
            .attr("stroke", "white");
    }

    if(node_data.data.probabilities){
        node.append("text")
            .attr("dy", "0.31em")
            .attr("y", 6)
            .attr("text-anchor", "middle")
            .text(d => JSON.stringify(d.data.probabilities))
            .clone(true).lower()
            .attr("stroke", "white");
    }

    if(node_data.data.logistic_model){

        const content = d3.create("svg:g");

        content.selectAll("g")
            .data(node_data)
            .enter()
            .append("text")
            .attr("dy", "0.31em")
            .attr("y", 6)
            .attr("text-anchor", "middle")
            .text(d => JSON.stringify(d.data.logistic_model))
            .clone(true).lower()
            .attr("stroke", "white");

        node.append(() => content.node());
    }

}