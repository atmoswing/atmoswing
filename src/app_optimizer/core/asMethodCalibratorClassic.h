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

#ifndef AS_METHOD_CALIBRATOR_CLASSIC_H
#define AS_METHOD_CALIBRATOR_CLASSIC_H

#include <asMethodCalibrator.h>

class asMethodCalibratorClassic : public asMethodCalibrator {
 public:
  asMethodCalibratorClassic();

  ~asMethodCalibratorClassic() override;

  void SetAsCalibrationPlus(bool val = true) {
    m_plus = val;
  }

 protected:
  bool Calibrate(asParametersCalibration &params) override;

 private:
  bool m_plus;
  int m_stepsLatPertinenceMap;
  int m_stepsLonPertinenceMap;
  int m_resizingIterations;
  bool m_proceedSequentially;

  void GetPlusOptions();

  bool DoPreloadData(asParametersCalibration &params);

  ParamExploration GetSpatialBoundaries(const asParametersCalibration &params, int iStep) const;

  void GetInitialAnalogNumber(asParametersCalibration &params, int iStep) const;

  void SetMinimalArea(asParametersCalibration &params, int iStep, const ParamExploration &explo) const;

  void GenerateRelevanceMapParameters(asParametersCalibration &params, int iStep, const ParamExploration &explo);

  void BalanceWeights(asParametersCalibration &params, int iStep) const;

  bool EvaluateRelevanceMap(const asParametersCalibration &params, asResultsDates &anaDatesPrevious,
                            asResultsParametersArray &resultsTested, int iStep);

  bool AssessDomainResizing(asParametersCalibration &params, asResultsDates &anaDatesPrevious,
                            asResultsParametersArray &resultsTested, int iStep, const ParamExploration &explo);

  bool AssessDomainResizingPlus(asParametersCalibration &params, asResultsDates &anaDatesPrevious,
                                asResultsParametersArray &resultsTested, int iStep, const ParamExploration &explo);

  bool GetDatesOfBestParameters(asParametersCalibration &params, asResultsDates &anaDatesPrevious, int iStep);

  void GetSpatialAxes(const asParametersCalibration &params, int iStep, const ParamExploration &explo, a1d &xAxis,
                      a1d &yAxis) const;

  void MoveWest(asParametersCalibration &params, const ParamExploration &explo, const a1d &xAxis, int iStep, int iPtor,
                int multipleFactor = 1) const;

  void MoveSouth(asParametersCalibration &params, const ParamExploration &explo, const a1d &yAxis, int iStep, int iPtor,
                 int multipleFactor = 1) const;

  void MoveEast(asParametersCalibration &params, const ParamExploration &explo, const a1d &xAxis, int iStep, int iPtor,
                int multipleFactor = 1) const;

  void MoveNorth(asParametersCalibration &params, const ParamExploration &explo, const a1d &yAxis, int iStep, int iPtor,
                 int multipleFactor = 1) const;

  void WidenEast(asParametersCalibration &params, const ParamExploration &explo, int iStep, int iPtor,
                 int multipleFactor = 1) const;

  void WidenNorth(asParametersCalibration &params, const ParamExploration &explo, int iStep, int iPtor,
                  int multipleFactor = 1) const;

  void ReduceEast(asParametersCalibration &params, const ParamExploration &explo, int iStep, int iPtor,
                  int multipleFactor = 1) const;

  void ReduceNorth(asParametersCalibration &params, const ParamExploration &explo, int iStep, int iPtor,
                   int multipleFactor = 1) const;
};

#endif
