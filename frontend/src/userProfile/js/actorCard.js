import React, {Component} from 'react';
import {Card} from 'antd';

const {Meta} = Card;

class ActorCard extends Component {
    render() {
        let prefix = process.env.PUBLIC_URL;
        return (
            <Card
                hoverable
                style={{width: 240}}
                cover={<img alt="Marilyn Monroe" src={prefix + "/source/images/actor/marilyn_monroe.jpg"}/>}
            >
                <Meta
                    title="Marilyn Monroe"
                    description="1926-1962"
                />
            </Card>
        )
    }
}

export default ActorCard