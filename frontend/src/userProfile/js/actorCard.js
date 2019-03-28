import React, {Component} from 'react';
import {Card} from 'antd';

const {Meta} = Card;

class ActorCard extends Component {
    constructor(props) {
        super(props);
        this.props.flag = "";
    }

    render() {
        let prefix = process.env.PUBLIC_URL;
        return (

            <div className={"6u "+this.props.flag} id="favoriteActor">
                <header>
                    <h2>最喜欢的演员是<span className="emphatic">章金莱</span></h2>
                    <span className="byline">看了<span className="emphatic">3</span>部他主演的电影，可以说是铁杆粉丝了</span>
                </header>
                <Card
                    hoverable
                    style={{width: 240}}
                    cover={<img alt="章金莱" src={prefix + "/source/images/actor/ZhangJinLai.jpg"}/>}
                >
                    <Meta
                        title="章金莱"
                        description="中国电视剧制作中心演员，代表作《西游记》、《猴娃》、《连城诀》等"
                    />
                </Card>
            </div>
        )
    }
}

export default ActorCard