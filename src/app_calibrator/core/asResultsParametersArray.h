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

#ifndef ASRESULTSPARAMETERSARRAY_H
#define ASRESULTSPARAMETERSARRAY_H

#include <asIncludes.h>
#include <asResults.h>

class asParametersScoring;

class asResultsParametersArray: public asResults
{
public:
    asResultsParametersArray();

    virtual ~asResultsParametersArray();

    /** Init
     * \param fileTag A tag to add to the file name
     */
    void Init(const wxString &fileTag);

    void Add(asParametersScoring params, float scoreCalib);

    void Add(asParametersScoring params, float scoreCalib, float scoreValid);

    void Clear();

    bool Print();

    void CreateFile();

    bool AppendContent();

protected:

    /** Build the result file path
     * \param fileTag A resulting file tag
     */
    void BuildFileName(const wxString &fileTag);

private:
    std::vector <asParametersScoring> m_Parameters;
    VectorFloat m_ScoresCalib;
    VectorFloat m_ScoresValid;
};

#endif // ASRESULTSPARAMETERSARRAY_H
