import React, {Component} from 'react';
import {Carousel, Divider, List} from "antd";
import {image_url} from "../../libs/toolFunctions";
import {getMoviePhotos} from "../../libs/getJsonData";

class CarouselImage extends Component {
    render() {
        const images = [];
        this.props.datas.forEach((d, i) => images.push(<img key={i} alt="" src={image_url(d)}/>));
        return (
            <Carousel effect='fade' autoplay>
                {images}
            </Carousel>
        )
    }
}

class PhotoGrid extends Component {
    render() {
        let i, j;
        let data = [];
        let totalPhotoNum = this.props.photos.length;
        if (totalPhotoNum === 0)
            return (<div/>);
        let num = this.props.carouselNum;
        for (i = 0; i < 9; i++) {
            data[i] = [];
            for (j = 0; j < num; j++) {
                let index = (i * num + j) % totalPhotoNum;
                data[i].push(this.props.photos[index]['cover']);
            }
        }

        return (
            <div>
                <List
                    grid={{gutter: 10, column: 2}}
                    dataSource={data.slice(0, 2)}
                    renderItem={(item, i) => (
                        <List.Item key={i}>
                            <CarouselImage datas={item}/>
                        </List.Item>
                    )}
                />

                <List
                    grid={{gutter: 10, column: 3}}
                    dataSource={data.slice(2, 5)}
                    renderItem={item => (
                        <List.Item>
                            <CarouselImage datas={item}/>
                        </List.Item>
                    )}
                />

                <List
                    grid={{gutter: 10, column: 4}}
                    dataSource={data.slice(5, 9)}
                    renderItem={item => (
                        <List.Item>
                            <CarouselImage datas={item}/>
                        </List.Item>
                    )}
                />
            </div>
        )

    }
}

PhotoGrid.defaultProps = {
    carouselNum: 6
};

class Stills extends Component {

    loadData(movieID) {
        getMoviePhotos(movieID, (data) => {
            this.setState({
                photosData: data,
                loadedFlag: true
            })
        });
    }

    constructor(props) {
        super(props);
        this.state = {
            photosData: [],
            loadedFlag: false,
        };
        this.loadData(this.props.movieID);
    }

    render() {
        if (!this.state.loadedFlag)
            return (<div/>);
        return (
            <div>
                <Divider>剧照</Divider>
                <PhotoGrid photos={this.state.photosData}/>
            </div>
        )
    }
}

export default Stills