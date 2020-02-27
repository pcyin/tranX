// Inspired by phpSyntaxTree by Andre Eisenbach <andre@ironcreek.net>
// https://code.google.com/p/phpsyntaxtree/

var pos = 0;
var data;
// root node
var root;
var currentNode;

function parse(textData) {
	// Reset
	pos = 0;
	currentNode = null;
	root = null;
	
	data = textData;
	
	// If this is a CCG tree keep only the categories and words
	if (data.match(/<T\s+(.*?)\s+[0-9]+\s+[0-9]+>/)) {
		// Replace any category-internal ()'s with [] to avoid confusing the parser
		// will be changed back later in TreeNode.toJSON()
		data = data.replace(/<T\s+(.*?)\s+[0-9]+\s+[0-9]+>/g, function(a, b){
		    return b.replace(/\[/g,"{").replace(/\]/g,"}").replace(/\(/g,"[").replace(/\)/g,"]");
		});
		data = data.replace(/<L\s+(.*?)\s+(.*?)\s+(.*?)\s+(.*?)\s+(.*?)>/g, function(a, b, c, d, e){
		    return b.replace(/\[/g,"{").replace(/\]/g,"}").replace(/\(/g,"[").replace(/\)/g,"]") + ' ' + e;
		});
	}
	
	// Clean up
	data = data.replace(/\s+/g, " ");
	data = data.replace(/\)\s+\(/g, ")(");
	data = data.replace(/\(\s+\(/g, "((");
	// Not sure why I have to do this twice... apparently 
	data = data.replace(/\)\s+\)/g, "))").replace(/\)\s+\)/g, "))");
	
	// Parse the string
	makeTree();
	root = currentNode;
	while (root.getParent() != null) root = root.getParent();
	return jQuery.parseJSON(root.toJSON());
}

function makeTree() {
    var token = getNextToken().trim();
	// console.log("First token: " + token);
	var parts = [2];
	
    while( token != "" && token != ")" ) {
        switch(token.charAt(0)) {
		case "(":
            token = token.substr(1, token.length - 1);
            var spaceAt = token.indexOf(" ");
			var childNode;
			
            if (spaceAt != -1) {
                parts[0] = token.substr(0, spaceAt);
                parts[1] = token.substr(spaceAt, token.length - spaceAt);
                childNode = new TreeNode(parts[0]);
				childNode.addChild(new TreeNode(parts[1]));
            } 
			else {
                childNode = new TreeNode(token);
            }
			
			if (currentNode) {
				// console.log("Current node: " + currentNode.toJSON());
				// console.log("Adding child" + childNode.toJSON());
				currentNode.addChild(childNode);
			}
			currentNode = childNode;
			makeTree();
            break;

        default:
            if (token != "") {
				currentNode.addChild(new TreeNode(token));
            }
            break;
        }
        token = getNextToken().trim();
    }
	if (token == ")") {
		if (currentNode.getParent()) {
			currentNode = currentNode.getParent();
			// console.log("Back to: " + currentNode.toJSON());
		}
	}
}

function getNextToken() {
    var gotToken = false;
    var token = "";
    var i = 0;

    if ((pos + 1) >= data.length) return "";

    while((pos + i) < data.length && !gotToken ) {
        var ch = data.charAt(pos + i);

        switch(ch) {
		case "(":
            if( i > 0 ) gotToken = true;
            else token += ch;
            break;

		case ")":
            if( i == 0 ) token += ch;
            gotToken = true;
            break;

        default:
            token += ch;
            break;
        }
        i++;
    }

    if (i > 1) pos += i - 1;
    else pos++;

    return token;
}

function TreeNode(label) {
	this.parent;
	this.name = label;
	this.children = [];
	
	this.addChild = function(child) {
		child.parent = this;
		this.children[this.children.length] = child;
	}
	
	this.getParent = function() {
		return this.parent;
	}
	
	this.getChild = function(index) {
		return this.children[index];
	}
	
	this.toJSON = function() {
		label = label.replace(/\[/g,"(").replace(/\]/g,")").replace(/{/g,"[").replace(/}/g,"]");
		var text = "{\"name\":" + "\"" + label + "\"";
		if (this.children.length != 0) {
			text += ",\"children\":[";
			for	(var index = 0; index < this.children.length; ++index) {
		    	text += this.children[index].toJSON() + ",";
			}
			//Remove the last comma
			text = text.substr(0, text.length-1);
			text += "]";
		}
		text += "}";
		return text;
	}
}
