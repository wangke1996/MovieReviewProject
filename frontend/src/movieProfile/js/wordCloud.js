import React, {Component} from 'react';
import {Chart, Geom, Tooltip, Coord, Shape} from 'bizcharts';
import DataSet from "@antv/data-set";

const colors = {POS: 'lime', NEU: 'orange', NEG: 'red'};

function getTextAttrs(cfg) {
    return {
        ...cfg.style,

        fillOpacity: cfg.opacity,
        fontSize: cfg.origin._origin.size,
        rotate: cfg.origin._origin.rotate,
        text: cfg.origin._origin.text,
        textAlign: 'center',
        fontFamily: cfg.origin._origin.font,
        fill: cfg.color,
        textBaseline: 'Alphabetic'
    };
}

// 给point注册一个词云的shape
Shape.registerShape('point', 'cloud', {
    drawShape(cfg, container) {
        const attrs = getTextAttrs(cfg);
        return container.addShape('text', {
            attrs: {
                ...attrs,
                x: cfg.x,
                y: cfg.y
            }
        });
    }
});
const data = [{
    "name": "China",
    "freq": 1383220000,
    "sentiment": "POS"
}, {
    "name": "India",
    "freq": 1316000000,
    "sentiment": "NEU"
}, {
    "name": "United States",
    "freq": 324982000,
    "sentiment": "NEG"
}, {
    "name": "Indonesia",
    "freq": 263510000,
    "sentiment": "POS"
}, {
    "name": "Brazil",
    "freq": 207505000,
    "sentiment": "NEU"
}, {
    "name": "Pakistan",
    "freq": 196459000,
    "sentiment": "NEG"
}, {
    "name": "Nigeria",
    "freq": 191836000,
    "sentiment": "POS"
}, {
    "name": "Bangladesh",
    "freq": 162459000,
    "sentiment": "NEG"
}];

export class WordCloud extends Component {
    state = {
        dv: null,
    };

    componentDidMount() {
        const dv = new DataSet.View().source(this.props.descriptions);
        const range = dv.range("freq");
        const min = range[0];
        const max = range[1];
        dv.transform({
            type: 'tag-cloud',
            fields: ["name", "freq"],
            // size: [500, 500],
            font: 'Verdana',
            padding: 0,
            timeInterval: 5000, // max execute time
            rotate() {
                let random = ~~(Math.random() * 4) % 4;
                if (random === 2) {
                    random = 0;
                }
                return random * 90; // 0, 90, 270
            },
            fontSize(d) {
                if (d.value) {
                    return ((d.value - min) / (max - min)) * (80 - 24) + 24;
                }
                return 0;
            }
        });
        this.setState({dv});


    }

    render() {
        const {dv} = this.state;
        if (!dv)
            return null;
        const scale = {
            x: {nice: false},
            y: {nice: false}
        };
        return <Chart data={dv} height={500} width={500} padding={0} scale={scale} onTooltipChange={(ev) => {
            ev.items[0].name = '频率';
            ev.items[1].name = '情感极性';
        }}>
            <Tooltip showTitle={false}/>
            <Coord reflect="y"/>
            <Geom
                type="point"
                position="x*y"
                color={["sentiment", sentiment => colors[sentiment]]}
                shape="cloud"
                tooltip='freq*sentiment'
            />
        </Chart>
    }
}
