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
 * Portions Copyright 2016-2019 Pascal Horton, University of Bern.
 */

#include <gtest/gtest.h>
#include <wx/filename.h>

#include "asAreaRegGrid.h"
#include "asPredictor.h"
#include "asTimeArray.h"

TEST(PredictorEcmwfEra5, GetCorrectPredictors) {
    asPredictor *predictor;

    predictor = asPredictor::GetInstance("ECMWF_ERA5", "pl/d", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Divergence);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA5", "pl/pv", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PotentialVorticity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA5", "pl/q", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SpecificHumidity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA5", "pl/r", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::RelativeHumidity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA5", "pl/t", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA5", "pl/u", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Uwind);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA5", "pl/v", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Vwind);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA5", "pl/vo", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Vorticity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA5", "pl/w", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::VerticalVelocity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA5", "pl/z", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Geopotential);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA5", "pl/x", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::ParameterUndefined);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA5", "single/d2m", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::DewpointTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA5", "single/msl", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Pressure);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA5", "single/sd", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SnowWaterEquivalent);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA5", "single/sst", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SeaSurfaceTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA5", "single/t2m", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA5", "single/tcw", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PrecipitableWater);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA5", "single/tcwv", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::WaterVapour);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA5", "single/u10", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Uwind);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA5", "single/v10", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Vwind);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA5", "single/tp", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Precipitation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA5", "single/cape", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::CAPE);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA5", "single/ie", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::MoistureFlux);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA5", "single/ssr", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA5", "single/ssrd", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA5", "single/str", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA5", "single/strd", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);
}
