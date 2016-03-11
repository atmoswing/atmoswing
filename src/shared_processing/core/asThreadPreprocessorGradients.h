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
#ifndef asThreadPreprocessorGradients_H
#define asThreadPreprocessorGradients_H

#include <asThread.h>
#include <asIncludes.h>

class asDataPredictor;


class asThreadPreprocessorGradients: public asThread
{
public:
    /** Default constructor */
    asThreadPreprocessorGradients(VArray2DFloat* gradients, std::vector < asDataPredictor* > predictors, int start, int end);
    /** Default destructor */
    virtual ~asThreadPreprocessorGradients();

    virtual ExitCode Entry();


protected:
private:
    VArray2DFloat* m_pGradients;
    std::vector < asDataPredictor* > m_pPredictors;
    int m_start;
    int m_End;

};

#endif // asThreadPreprocessorGradients_H
