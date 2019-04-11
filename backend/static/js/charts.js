var test = function () {
    console.log("test")
};
var drawBars = function (place, userData,x_col,y_col) {
    var margin = { top: 50, right: 60, bottom: 30, left: 80 },
        width = 700 - margin.left - margin.right,
        height = 400 - margin.top - margin.bottom;

    var x = d3.scaleLinear()
    //.rangeRound([0, width]);
    var y = d3.scaleBand()
    //.rangeRoundBands([0, height], .5, .3);

    var xAxis = d3.axisTop()
        .scale(x)

    var yAxis = d3.axisLeft()
        .scale(y)
    
    var classList = d3.selectAll(place)._groups[0]
    var last_element = classList[classList.length-1]

    var svg = d3.select(last_element).append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
    
    var data =
        [
            { "letter": "A", "frequency": 4167 },
            { "letter": "B", "frequency": 6167 },
            { "letter": "C", "frequency": 3167 },
            { "letter": "D", "frequency": 5127 },
            { "letter": "E", "frequency": 2167 },
            { "letter": "F", "frequency": 8167 },
            { "letter": "G", "frequency": 3167 },
            { "letter": "H", "frequency": 9167 }
        ];

    data = userData
    y.domain(data.map(function (d) { return d.letter; })).range([0, height]);
    x.domain([0, d3.max(data, function (d) { return d.frequency; })]).range([0, width]);

    svg.append("text")
                .attr("class", "x-axis-label")
                .attr("text-anchor", "middle")
                .attr("x", width/2)
                .attr("y", height-1)
                .text(x_col+"-" + y_col);

    svg.append("g")
        .attr("class", "x axis")
        .call(xAxis)
        .append("text")
        .text("Frequency")
        .style("font", "10px sans-serif");

    svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(" + 0 + ",0)")
        .call(xAxis);


    svg.append("g")
        .attr("class", "y axis")
        .attr("transform", "translate(0 ,0)")
        .call(yAxis);


    svg.selectAll(".bar")
        .data(data)
        .enter()
        .append("rect")
        .attr("class", "bar")
        .attr("x", 0)
        .attr("width", function (d) { return x(d.frequency); })
        .attr("y", function (d) { return y(d.letter); })
        .attr("height", y.step() / 2)
        .on("mouseover", hyper_mouse);

    // var text = svg.selectAll(text)
    //     .data(data)
    //     .enter()
    //     .append("text")
    //     .attr("x", function (d) { return x(d.frequency); })
    //     .attr("y", function (d) { return y(d.letter); })
    //     .attr("fill","black")
    //     .attr("dx", 0)
    //     .attr("dy", "1em")
    //     .style("font", "10px sans-serif")
    //     .attr("text-anchor", "begin")
    //     .text(function (d) {
    //         return d.frequency
    //     });

    // d3.select("input")
    //     .on("change", change);

    // var sortTimeout = setTimeout(function () {
    //     d3.select("input").property("checked", false).each(change);
    // }, 2000);

    function hyper_mouse(d) {
        $('.bar').hover(function () {
            $(this).css('fill-opacity', "0.7");
        }, function () {
            $(this).css('fill-opacity', "0.9");
        });
    }

    function change() {
        clearTimeout(sortTimeout);
        // Copy-on-write since tweens are evaluated after a delay.
        var x0 = y.domain(data.sort(this.checked
            ? function (a, b) { return b.frequency - a.frequency; }
            : function (a, b) { return d3.ascending(a.letter, b.letter); })
            .map(function (d) { return d.letter; }))
            .copy();

        svg.selectAll(".bar")
            .sort(function (a, b) { return x0(a.letter) - x0(b.letter); });



        var transition = svg.transition().duration(750),
            delay = function (d, i) { return i * 50; };

        transition.selectAll(".bar")
            .delay(delay)
            .attr("y", function (d) { return x0(d.letter); });

        transition.select(".y.axis")
            .call(yAxis)
            .selectAll("g")
            .delay(delay);
    }
}

var drawLines = function(place,userData,x_col,y_col){

    var margin = { top: 50, right: 60, bottom: 30, left: 40 },
            width = 600 - margin.left - margin.right,
            height = 400 - margin.top - margin.bottom;

        var x = d3.scaleBand()
            //.rangeRoundBands([0, width]);

        var y = d3.scaleLinear()
            //.rangeRound([0, height]);


        var xAxis = d3.axisBottom()
            .scale(x)

        var yAxis = d3.axisLeft()
            .scale(y);

        var classList = d3.selectAll(place)._groups[0]
        var last_element = classList[classList.length-1]
        
        var svg = d3.select(last_element).append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")");


        var data =
            [
                { "letter": "A", "frequency": 4167 },
                { "letter": "B", "frequency": 6167 },
                { "letter": "C", "frequency": 3167 },
                { "letter": "D", "frequency": 5127 },
                { "letter": "E", "frequency": 2167 },
                { "letter": "F", "frequency": 8167 },
                { "letter": "G", "frequency": 3167 },
                { "letter": "H", "frequency": 9167 }
            ];

        data = userData
        temp_x = data.map(function (d) { return d.letter; });
        x.domain(temp_x).range([0,width]);
        max = d3.max(data, function (d) { return d.frequency; });
        y.domain([0, max]).range([height,0]);

        
        svg.append("text")
                .attr("class", "x-axis-label")
                .attr("text-anchor", "middle")
                .attr("x", width/2)
                .attr("y", 0)
                .text(x_col+"-" + y_col);

        svg.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0," + height + ")")
            .call(xAxis);


        svg.append("g")
            .attr("class", "y axis")
            .attr("transform", "translate(0,0)")
            .call(yAxis);

         var linePath = d3.line()
                 .x(function (d, i) {
                     return x(d.letter)-10
                 })
                 .y(function (d) {
                     return y(d.frequency)-margin.top
                 });


        svg.append("g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
            .append("path")
            .attr("d", linePath(data))
            .attr('stroke', 'black')
            .attr('stroke-width', 1)
            .attr("fill", "none")
}