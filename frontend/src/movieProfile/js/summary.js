import React, {Component} from 'react';
import {image_url} from "../../libs/toolFunctions";
import {Tag} from "antd";
import '../css/summary.css'

class MovieTags extends Component {
    render() {
        const presetColors = ['magenta', 'red', 'volcano', 'orange', 'gold', 'lime', 'green', 'cyan', 'blue', 'geekblue', 'purple']
        const tagElements = [];
        this.props.tags.forEach((d, i) => {
            let color = presetColors[i % presetColors.length];
            tagElements.push(<Tag color={color}>{d}</Tag>);
        });
        return (
            <span className="MovieTags">
                {tagElements}
            </span>
        );
    }
}

class Summary extends Component {
    render() {
        const data = this.props.data;
        return (
            <div>
                <header>
                    <h2>{data['title']}({data['year']})</h2>
                    <MovieTags tags={data['tags']}/>
                </header>
                <div className="row">
                    <div className="4u">
                        <a href={data['images']['large']} className="image">
                            <img src={image_url(data['images']['large'])} alt=""/>
                        </a>
                    </div>
                    <div className="8u summary">
                        <header><h3>剧情简介</h3></header>
                        <p>{data['summary']}</p>
                    </div>
                </div>
            </div>
        );
    }
}

export default Summary;