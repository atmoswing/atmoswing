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
 * The Original Software is AtmoSwing. The Initial Developer of the 
 * Original Software is Pascal Horton of the University of Lausanne. 
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 * Portions Copyright 2013 Pascal Horton, Terr@num.
 */
 
#ifndef ASRESULTSANALOGSSCORESMAP_H
#define ASRESULTSANALOGSSCORESMAP_H

#include <asIncludes.h>
#include <asResults.h>

class asParametersCalibration;
class asParametersScoring;


class asResultsAnalogsScoresMap: public asResults
{
public:

    /** Default constructor */
    asResultsAnalogsScoresMap();

    /** Default destructor */
    virtual ~asResultsAnalogsScoresMap();

    /** Init
     * \param params The parameters structure
     */
    void Init(asParametersScoring &params);

    /** Add data
     * \param params The parameters structure
     * \param score The score value
     * \return True on success
     */
    bool Add(asParametersScoring &params, float score);

    /** Make the map on data basis
     * \return True on success
     */
    bool MakeMap();

    /** Save the result file
     * \param AlternateFilePath An optional file path
     * \return True on success
     */
    bool Save(asParametersCalibration &params, const wxString &AlternateFilePath = wxEmptyString);

protected:

    /** Build the result file path
     * \param params The parameters structure
     */
    void BuildFileName(asParametersScoring &params);

private:
    Array1DFloat m_MapLon;
    Array1DFloat m_MapLat;
    Array1DFloat m_MapLevel;
    VArray2DFloat m_MapScores; //!< Member variable "m_Scores"
    VectorFloat m_Scores; //!< Member variable "m_Scores".
    VectorFloat m_Lon; //!< Member variable "m_Lon".
    VectorFloat m_Lat; //!< Member variable "m_Lat".
    VectorFloat m_Level; //!< Member variable "m_Level".
};

#endif // ASRESULTSANALOGSSCORESMAP_H
