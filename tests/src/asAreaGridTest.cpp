/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can read the License at http://opensource.org/licenses/CDDL-1.0
 * See the License for the specific language governing permissions
 * and limitations under the License.
 *
 * When distributing Covered Code, include this CDDL Header Notice in
 * each file and include the License file (licence.txt). If applicable,
 * add the following below this CDDL Header, with the fields enclosed
 * by brackets [] replaced by your own identifying information:
 * "Portions Copyright [year] [name of copyright owner]"
 *
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#include <gtest/gtest.h>

#include "asAreaRegGrid.h"

TEST(AreaGrid, CheckConsistency) {
    double xMin = -5;
    int xPtsNb = 11;
    double yMin = 30;
    int yPtsNb = 5;
    double step = 2.5;
    wxString gridType = "Regular";
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    EXPECT_DOUBLE_EQ(355, area->GetCornerUL().x);
    EXPECT_DOUBLE_EQ(355, area->GetCornerLL().x);
    EXPECT_DOUBLE_EQ(20, area->GetCornerUR().x);
    EXPECT_DOUBLE_EQ(20, area->GetCornerLR().x);
    wxDELETE(area);
}

TEST(AreaGrid, CheckConsistencyException) {
    wxLogNull logNo;

    double xMin = 10;
    int xPtsNb = 1;
    double yMin = 40;
    int yPtsNb = -5;
    double step = 2.5;
    wxString gridType = "Regular";
    asAreaGrid *area = nullptr;
    ASSERT_THROW(area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step), std::exception);
    wxDELETE(area);
}

TEST(AreaGrid, GetBoundsSplitted) {
    double xMin = -10;
    int xPtsNb = 13;
    double yMin = 30;
    int yPtsNb = 5;
    double step = 2.5;
    wxString gridType = "Regular";
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    a1d lons = a1d::LinSpaced(145, 0.0, 360.0);
    a1d lats = a1d::LinSpaced(73, -90.0, 90.0);
    area->InitializeAxes(lons, lats);

    EXPECT_DOUBLE_EQ(-10, area->GetXmin());
    EXPECT_DOUBLE_EQ(20, area->GetXmax());
    EXPECT_DOUBLE_EQ(30, area->GetYmin());
    EXPECT_DOUBLE_EQ(40, area->GetYmax());
    wxDELETE(area);
}

TEST(AreaGrid, GetUYaxisSize) {
    double xMin = -40;
    int xPtsNb = 21;
    double yMin = 30;
    int yPtsNb = 5;
    double step = 2.5;
    wxString gridType = "Regular";
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    a1d lons = a1d::LinSpaced(145, 0.0, 360.0);
    a1d lats = a1d::LinSpaced(73, -90.0, 90.0);
    area->InitializeAxes(lons, lats);

    EXPECT_EQ(21, area->GetXaxisPtsnb());
    EXPECT_EQ(5, area->GetYaxisPtsnb());
    wxDELETE(area);
}

TEST(AreaGrid, GetUYaxisSizeStepLon) {
    double xMin = -40;
    int xPtsNb = 11;
    double yMin = 30;
    int yPtsNb = 5;
    double xStep = 5;
    double yStep = 2.5;
    wxString gridType = "Regular";
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb, yStep);

    a1d lons = a1d::LinSpaced(145, 0.0, 360.0);
    a1d lats = a1d::LinSpaced(73, -90.0, 90.0);
    area->InitializeAxes(lons, lats);

    EXPECT_EQ(11, area->GetXaxisPtsnb());
    EXPECT_EQ(5, area->GetYaxisPtsnb());
    wxDELETE(area);
}

TEST(AreaGrid, GetUYaxisSizeStepLonMoved) {
    double xMin = -7.5;
    int xPtsNb = 4;
    double yMin = 30;
    int yPtsNb = 5;
    double xStep = 5;
    double yStep = 2.5;
    wxString gridType = "Regular";
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb, yStep);

    a1d lons = a1d::LinSpaced(145, 0.0, 360.0);
    a1d lats = a1d::LinSpaced(73, -90.0, 90.0);
    area->InitializeAxes(lons, lats);

    EXPECT_EQ(4, area->GetXaxisPtsnb());
    EXPECT_EQ(5, area->GetYaxisPtsnb());
    wxDELETE(area);
}

TEST(AreaGrid, GetUYaxisLimits) {
    double xMin = -10;
    int xPtsNb = 5;
    double yMin = 30;
    int yPtsNb = 5;
    double xStep = 5;
    double yStep = 2.5;
    wxString gridType = "Regular";
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb, yStep);

    a1d lons = a1d::LinSpaced(145, 0.0, 360.0);
    a1d lats = a1d::LinSpaced(73, -90.0, 90.0);
    area->InitializeAxes(lons, lats);

    EXPECT_DOUBLE_EQ(-10, area->GetXaxisStart());
    EXPECT_DOUBLE_EQ(30, area->GetYaxisStart());
    EXPECT_DOUBLE_EQ(10, area->GetXaxisEnd());
    EXPECT_DOUBLE_EQ(40, area->GetYaxisEnd());
    wxDELETE(area);
}

