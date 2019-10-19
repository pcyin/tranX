// Inspired by "D3.js Drag and Drop Zoomable Tree" by Rob Schmuecker <robert.schmuecker@gmail.com>
// https://gist.github.com/robschmuecker/7880033

function d3Tree(treeData, hyp_id, viewerWidth = 1020) {
    // panning variables
    var panSpeed = 200;
    // Misc. variables
    var i = 0;
    var duration = 450;
    var root;
	
    // size of the diagram
	//var pageWidth = $(document).width();
    //var viewerWidth = pageWidth - (0.2 * pageWidth);
    var viewerHeight = 500;

    var tree = d3.layout.tree()
        .size([viewerWidth-20, viewerHeight]);

    // define a d3 diagonal projection for use by the node paths later on.
    var diagonal = d3.svg.diagonal()
        .projection(function(d) {
            return [d.x, d.y];
        });
		
	// Can be used to draw the links between nodes instead of the diagonal
	// TODO Doesn't work with the collapse/expand transition
	//function straightLine(d) {
	//	return "M" + d.source.x + "," + d.source.y + "L" + d.target.x + "," + d.target.y;
	//}

    // A recursive helper function for performing some setup by walking through all nodes
    function visit(parent, visitFn, childrenFn) {
        if (!parent) return;

        visitFn(parent);

        var children = childrenFn(parent);
        if (children) {
            var count = children.length;
            for (var i = 0; i < count; i++) {
                visit(children[i], visitFn, childrenFn);
            }
        }
    }

    // TODO: Pan function, can be better implemented.
    function pan(domNode, direction) {
        var speed = panSpeed;
        if (panTimer) {
            clearTimeout(panTimer);
            translateCoords = d3.transform(svgGroup.attr("transform"));
            if (direction == 'left' || direction == 'right') {
                translateX = direction == 'left' ? translateCoords.translate[0] + speed : translateCoords.translate[0] - speed;
                translateY = translateCoords.translate[1];
            } else if (direction == 'up' || direction == 'down') {
                translateX = translateCoords.translate[0];
                translateY = direction == 'up' ? translateCoords.translate[1] + speed : translateCoords.translate[1] - speed;
            }
            scaleX = translateCoords.scale[0];
            scaleY = translateCoords.scale[1];
            scale = zoomListener.scale();
            svgGroup.transition().attr("transform", "translate(" + translateX + "," + translateY + ")scale(" + scale + ")");
            d3.select(domNode).select('g.node').attr("transform", "translate(" + translateX + "," + translateY + ")");
            zoomListener.scale(zoomListener.scale());
            zoomListener.translate([translateX, translateY]);
            panTimer = setTimeout(function() {
                pan(domNode, speed, direction);
            }, 50);
        }
    }

    // Define the zoom function for the zoomable tree
    function zoom() {
        svgGroup.attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
    }


    // define the zoomListener which calls the zoom function on the "zoom" event constrained within the scaleExtents
    var zoomListener = d3.behavior.zoom().scaleExtent([0.1, 3]).on("zoom", zoom);
	
	// remove the previous svg if there
	d3.select(`#tree_container_hyp${hyp_id} svg`).remove();
	
    // define the baseSvg, attaching a class for styling and the zoomListener
    var baseSvg = d3.select(`#tree_container_hyp${hyp_id}`).append("svg")
        .attr("width", viewerWidth)
        .attr("height", viewerHeight)
        .attr("class", "overlay")
        .call(zoomListener)
		.on("dblclick.zoom", null);

	// The arrowmarker to be appended at the end of each path
	// TODO Looks terrible (not currently used)
	baseSvg.append("marker")
		.attr("id", "markerArrow")
		.attr("markerWidth", 4)
		.attr("markerHeight", 4)
		.attr("refY","2")
		.attr("refX", "10")
		.attr("orient", "auto")
		.append("polygon")
		.attr("points", "0,0 4,2 0,4")
		.attr("style", "fill: #ccc");

    // Helper functions for collapsing and expanding nodes.
    function collapse(d) {
        if (d.children) {
            d._children = d.children;
            d._children.forEach(collapse);
            d.children = null;
        }
    }

    function expand(d) {
        if (d._children) {
            d.children = d._children;
            d.children.forEach(expand);
            d._children = null;
        }
    }

    var overCircle = function(d) {
        selectedNode = d;
        updateTempConnector();
    };
    var outCircle = function(d) {
        selectedNode = null;
        updateTempConnector();
    };

    // Toggle children function
    function toggleChildren(d) {
        if (d.children) {
            d._children = d.children;
            d.children = null;
        } else if (d._children) {
            d.children = d._children;
            d._children = null;
        }
        return d;
    }

    // Toggle children on click.
    function click(d) {
        if (d3.event.defaultPrevented) return; // click suppressed
        d = toggleChildren(d);
        update(d);
    }

    function update(source) {
        // Compute the new height, function counts total children of root node and sets tree height accordingly.
        // This prevents the layout looking squashed when new nodes are made visible or looking sparse when nodes are removed
        // This makes the layout more consistent.
        var levelHeight = [1];
        var childCount = function(level, n) {
            if (n.children && n.children.length > 0) {
                if (levelHeight.length <= level + 1) levelHeight.push(0);

                levelHeight[level + 1] += n.children.length;
                n.children.forEach(function(d) {
                    childCount(level + 1, d);
                });
            }
        };
		childCount(0, root);
		var maxLevel = levelHeight.length+2;

        // Compute the new tree layout.
        var nodes = tree.nodes(root).reverse(),
            links = tree.links(nodes);

        // Set heights between levels based on maxLevel.
        nodes.forEach(function(d) {
            d.y = (d.depth * (viewerHeight/(maxLevel)));
        });

        // Update the nodes…
        node = svgGroup.selectAll("g.node")
            .data(nodes, function(d) {
                return d.id || (d.id = ++i);
            });

        // Enter any new nodes at the parent's previous position.
        var nodeEnter = node.enter().append("g")
            .attr("class", "node")
            .attr("transform", function(d) {
                return "translate(" + source.x0 + "," + source.y0 + ")";
            })
            .on('click', click);

		nodeEnter.append("rect")
			.attr('class', 'nodeRect')
			// Size of the rectangle/2
			.attr("x", function(d){return -(d.name.length*5+10)/2})
			.attr("y", -10)
			.attr("width", 0)
			.attr("height", 0)
			.style("fill", function(d) {
			    return d._children ? "lightsteelblue" : "#fff";
		});

        nodeEnter.append("text")
            .attr("y", 0)
            .attr("dy", ".35em")
            .attr('class', 'nodeText')
            .attr("text-anchor", "middle")
            .text(function(d) {
                return d.name;
            })
            .style("fill-opacity", 0);

        // Update the text to reflect whether node has children or not.
        node.select('text')
            .attr("y", 0)
            .attr("text-anchor", "middle")
            .text(function(d) {
                return d.name;
            });

        node.select("rect.nodeRect")
            .attr("width", function(d) {
				// Adjust the size of the square according to the label
                return d.children || d._children ? d.name.length*5+10 : 0;
            })
            .attr("height", function(d) {
                return d.children || d._children ? 20 : 0;
            })
            .style("fill", function(d) {
                return d._children ? "lightsteelblue" : "#fff";
            });

        // Transition nodes to their new position.
        var nodeUpdate = node.transition()
            .duration(duration)
            .attr("transform", function(d) {
                return "translate(" + d.x + "," + d.y + ")";
            });

        // Fade the text in
        nodeUpdate.select("text")
            .style("fill-opacity", 1);

        // Transition exiting nodes to the parent's new position.
        var nodeExit = node.exit().transition()
            .duration(duration)
            .attr("transform", function(d) {
                return "translate(" + source.x + "," + source.y + ")";
            })
            .remove();

        nodeExit.select("circle")
            .attr("r", 0);

        nodeExit.select("text")
            .style("fill-opacity", 0);

        // Update the links…
        var link = svgGroup.selectAll("path.link")
            .data(links, function(d) {
                return d.target.id;
            });

        // Enter any new links at the parent's previous position.
        link.enter().insert("path", "g")
            .attr("class", "link")
			//TODO MARKERS LOOK TERRIBLE
			// .attr("marker-end", "url(#markerArrow)")
            .attr("d", function(d) {
                var o = {x: source.x, y: source.y};
                return diagonal({source: o,target: o});
            });
			// TODO doesn't work with the transition
			// .attr("d", straightLine);

        // Transition links to their new position.
        link.transition()
            .duration(duration)
            .attr("d", diagonal);
			// TODO doesn't work with the transition
            // .attr("d", straightLine);

        // Transition exiting nodes to the parent's new position.
        link.exit().transition()
            .duration(duration)
            .attr("d", function(d) {
                var o = {x: source.x, y: source.y};
                return diagonal({source: o,target: o});
            })
			// TODO doesn't work with the transition
			// .attr("d", straightLine)
            .remove();

        // Stash the old positions for transition.
        nodes.forEach(function(d) {
            d.x0 = d.x;
            d.y0 = d.y;
        });
    }

    // Append a group which holds all nodes and which the zoom Listener can act upon.
    var svgGroup = baseSvg.append("g");

    // Define the root
    root = treeData;
    root.x0 = viewerWidth / 2;
    root.y0 = 0;

    // Layout the tree initially and center on the root node.
    update(root);
    d3.select('g').attr("transform", "translate(0,20)");
}