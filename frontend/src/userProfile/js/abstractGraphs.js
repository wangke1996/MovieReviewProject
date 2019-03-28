import React, {Component} from 'react';
import RateDistribution from './rateDistribution';
import FavoriteType from './favoriteType';
import ReviewList from './reviewList';
import AgeDistribution from './ageDistribution';
import ActorCard from './actorCard';
import TotalNum from './totalNum';

class AbstractGraphs extends Component {

    constructor(props) {
        super(props);
        this.state = {
            flagTotalNum: true,
            flagRateDistribution: true,
            flagFavoriteType: true,
            flagReviewList: true,
            flagAgeDistribution: true,
            flagFavoriteActor: true
        };
    }

    render() {
        return (
            <div className="AbstractGraph align-center">
                <div className="row">
                    <TotalNum flag={this.state.flagTotalNum}/>
                    <RateDistribution flag={this.state.flagRateDistribution}/>
                </div>
                <div className="row">
                    <FavoriteType flag={this.state.flagFavoriteType}/>
                    <ReviewList flag={this.state.flagReviewList}/>
                </div>
                <div className="row">
                    <AgeDistribution flag={this.state.flagAgeDistribution}/>
                    <ActorCard flag={this.state.flagFavoriteActor}/>
                </div>
            </div>
        )
    }
}

export default AbstractGraphs