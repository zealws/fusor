"""Module for modifying fusion objects."""

import logging
import re
from typing import Annotated

from bioutils.accessions import coerce_namespace
from civicpy.civic import MolecularProfile
from cool_seq_tool.app import CoolSeqTool
from cool_seq_tool.schemas import CoordinateType, Strand
from ga4gh.core import ga4gh_identify
from ga4gh.core.models import MappableConcept
from ga4gh.vrs import models
from ga4gh.vrs.models import (
    LiteralSequenceExpression,
    SequenceLocation,
    SequenceReference,
    sequenceString,
)
from gene.database import AbstractDatabase as GeneDatabase
from gene.database import create_db
from gene.query import QueryHandler
from gene.schemas import CURIE_REGEX
from pydantic import StringConstraints, ValidationError

from fusor.exceptions import FUSORParametersException, IDTranslationException
from fusor.models import (
    AnchoredReads,
    Assay,
    AssayedFusion,
    AssayedFusionElement,
    BaseStructuralElement,
    BreakpointCoverage,
    CategoricalFusion,
    CategoricalFusionElement,
    CausativeEvent,
    DomainStatus,
    FunctionalDomain,
    Fusion,
    FusionType,
    GeneElement,
    LinkerElement,
    MultiplePossibleGenesElement,
    RegulatoryClass,
    RegulatoryElement,
    StructuralElementType,
    TemplatedSequenceElement,
    TranscriptSegmentElement,
    UnknownGeneElement,
)
from fusor.nomenclature import generate_nomenclature
from fusor.tools import get_error_message, translate_identifier

_logger = logging.getLogger(__name__)


class FUSOR:
    """Class for modifying fusion objects."""

    def __init__(
        self,
        gene_database: GeneDatabase | None = None,
        cool_seq_tool: CoolSeqTool | None = None,
    ) -> None:
        """Initialize FUSOR class.

        :param cool_seq_tool: Cool-Seq-Tool instance
        :param gene_database: gene normalizer database instance
        """
        self._gene_database = gene_database
        self._gene_normalizer = None

        self._cool_seq_tool = cool_seq_tool
        self._seqrepo = None

    @property
    def gene_database(self):
        """
        :return: the gene-normalizer's connection to the UTA database
        """
        if not self._gene_database:
            self._gene_database = create_db()
        return self._gene_database

    @property
    def gene_normalizer(self) -> QueryHandler:
        """
        :return: the QueryHandler client for gene-normalizer
        """
        if not self._gene_normalizer:
            self._gene_normalizer = QueryHandler(self.gene_database)
        return self._gene_normalizer

    @property
    def cool_seq_tool(self) -> CoolSeqTool:
        """
        :return: the client for CoolSeqTool
        """
        if not self._cool_seq_tool:
            self._cool_seq_tool = CoolSeqTool()
        return self._cool_seq_tool

    @property
    def seqrepo(self):
        """
        :return: the client for SeqRepo, accessed through cool-seq-tool
        """
        if not self._seqrepo:
            self._seqrepo = self.cool_seq_tool.seqrepo_access.sr
        return self._seqrepo



    @staticmethod
    def _contains_element_type(kwargs: dict, elm_type: StructuralElementType) -> bool:
        """Check if fusion contains element of a specific type. Helper method for
        inferring fusion type.

        :param kwargs: keyword args given to fusion method
        :param elm_type: element type to match
        :return: True if at least one element of given type is found, False otherwise.
        """
        for c in kwargs["structure"]:
            if (isinstance(c, dict) and c.get("type") == elm_type) or (
                isinstance(c, BaseStructuralElement) and c.type == elm_type
            ):
                return True
        return False

    def fusion(self, fusion_type: FusionType | None = None, **kwargs) -> Fusion:
        """Construct Fusion object.

        Fusion type (assayed vs categorical) can be inferred based on provided kwargs,
        assuming they can sufficiently discriminate the type.

        :param fusion_type: explicitly specify fusion type. Unnecessary if providing
            fusion object in keyword args that includes ``type`` attribute.
        :return: constructed fusion object if successful
        :raise: FUSORParametersException if fusion type unable to be determined,
            or if incorrect fusion parameters are provided
        """
        # try explicit type param
        explicit_type = kwargs.get("type")
        if not fusion_type and explicit_type:
            if explicit_type in FusionType.values():
                fusion_type = explicit_type
                kwargs.pop("type")
            else:
                msg = f"Invalid type parameter: {explicit_type}"
                raise FUSORParametersException(msg)
        fusion_fn = None
        if fusion_type:
            if fusion_type == FusionType.CATEGORICAL_FUSION:
                fusion_fn = self.categorical_fusion
            elif fusion_type == FusionType.ASSAYED_FUSION:
                fusion_fn = self.assayed_fusion
            else:
                msg = f"Invalid fusion_type parameter: {fusion_type}"
                raise FUSORParametersException(msg)
        else:
            # try to infer from provided attributes
            categorical_attributes = any(
                [
                    "critical_functional_domains" in kwargs,
                    "civic_molecular_profiles" in kwargs,
                    self._contains_element_type(
                        kwargs, StructuralElementType.MULTIPLE_POSSIBLE_GENES_ELEMENT
                    ),
                ]
            )
            assayed_attributes = any(
                [
                    "causative_event" in kwargs,
                    "assay" in kwargs,
                    self._contains_element_type(
                        kwargs, StructuralElementType.UNKNOWN_GENE_ELEMENT
                    ),
                ]
            )
            if categorical_attributes and assayed_attributes:
                msg = "Received conflicting attributes"
                raise FUSORParametersException(msg)
            if categorical_attributes and not assayed_attributes:
                fusion_fn = self.categorical_fusion
            elif assayed_attributes and not categorical_attributes:
                fusion_fn = self.assayed_fusion
        if fusion_fn is None:
            msg = "Unable to determine fusion type"
            raise FUSORParametersException(msg)
        try:
            return fusion_fn(**kwargs)
        except TypeError as e:
            msg = f"Unable to construct fusion with provided args: {e}"
            raise FUSORParametersException(msg) from e

    @staticmethod
    def categorical_fusion(
        structure: list[CategoricalFusionElement],
        regulatory_element: RegulatoryElement | None = None,
        critical_functional_domains: list[FunctionalDomain] | None = None,
        reading_frame_preserved: bool | None = None,
        civic_molecular_profiles: list[MolecularProfile] | None = None,
    ) -> CategoricalFusion:
        """Construct a CategoricalFusion object

        :param structure: elements constituting the fusion
        :param regulatory_element: affected regulatory element
        :param critical_functional_domains: lost or preserved functional domains
        :param reading_frame_preserved: ``True`` if reading frame is preserved.
            ``False`` otherwise
        :param civic_molecular_profiles: A list of MolecularProfile objects
        :return: CategoricalFusion if construction successful
        :raise: FUSORParametersException if given incorrect fusion properties
        """
        try:
            fusion = CategoricalFusion(
                structure=structure,
                criticalFunctionalDomains=critical_functional_domains,
                readingFramePreserved=reading_frame_preserved,
                regulatoryElement=regulatory_element,
                civic_molecular_profiles=civic_molecular_profiles,
            )
        except ValidationError as e:
            error_message = get_error_message(e)
            raise FUSORParametersException(error_message) from e
        return fusion

    @staticmethod
    def assayed_fusion(
        structure: list[AssayedFusionElement],
        causative_event: CausativeEvent | None = None,
        assay: Assay | None = None,
        regulatory_element: RegulatoryElement | None = None,
        reading_frame_preserved: bool | None = None,
    ) -> AssayedFusion:
        """Construct an AssayedFusion object.

        :param structure: elements constituting the fusion
        :param causative_event: event causing the fusion
        :param assay: how knowledge of the fusion was obtained
        :param regulatory_element: affected regulatory elements
        :param reading_frame_preserved: ``True`` if reading frame is preserved.
            ``False`` otherwise.
        :return: Tuple containing optional ``AssayedFusion`` if construction successful,
            and any relevant validation warnings
        """
        try:
            fusion = AssayedFusion(
                structure=structure,
                regulatoryElement=regulatory_element,
                causativeEvent=causative_event,
                assay=assay,
                readingFramePreserved=reading_frame_preserved,
            )
        except ValidationError as e:
            error_message = get_error_message(e)
            raise FUSORParametersException(error_message) from e
        return fusion

    async def transcript_segment_element(
        self,
        tx_to_genomic_coords: bool = True,
        use_minimal_gene: bool = True,
        seq_id_target_namespace: str | None = None,
        coverage: BreakpointCoverage | None = None,
        reads: AnchoredReads | None = None,
        **kwargs,
    ) -> tuple[TranscriptSegmentElement | None, list[str] | None]:
        """Create TranscriptSegmentElement.

        :param tx_to_genomic_coords: ``True`` if going from transcript to genomic
            coordinates. ``False`` if going from genomic to transcript exon coordinates.
        :param use_minimal_gene: `True` if minimal gene object
            (``id``, ``label``) will be used. ``False`` if
            gene-normalizer's entire gene object will be used
        :param seq_id_target_namespace: If want to use digest for ``sequence_id``, set
            this to the namespace you want the digest for. Otherwise, leave as ``None``.
        :param coverage: The read coverage located near the specified breakpoint
        :param reads: The read data for the specified breakpoint
        :param kwargs:
            If ``tx_to_genomic_coords``, possible key word arguments:

                (From `cool_seq_tool.tx_segment_to_genomic <https://coolseqtool.readthedocs.io/stable/reference/api/mappers/cool_seq_tool.mappers.exon_genomic_coords.html>`_)

                * **gene** (``str | None = None``)
                * **transcript** (``str | None = None``)
                * **exon_start** (``int | None = None``)
                * **exon_start_offset**: Optional[int] = 0
                * **exon_end**: Optional[int] = None
                * **exon_end_offset**: (``Optional[int] = 0``)

            else:

                (From `cool_seq_tool.genomic_to_tx_segment <https://coolseqtool.readthedocs.io/stable/reference/api/mappers/cool_seq_tool.mappers.exon_genomic_coords.html>`_)

                * **genomic_ac**: (``str``)
                * **seg_start_genomic**: (``Optional[int] = None``)
                * **seg_end_genomic**: (``Optional[int] = None``)
                * **transcript**: (``Optional[str] = None``)
                * **gene**: (``Optional[str] = None``)

        :return: Transcript Segment Element, warning
        """
        if tx_to_genomic_coords:
            data = await self.cool_seq_tool.ex_g_coords_mapper.tx_segment_to_genomic(
                **kwargs
            )
        else:
            if "genomic_ac" in kwargs and kwargs.get("genomic_ac") is None:
                msg = (
                    "`genomic_ac` is required when going from genomic to"
                    " transcript exon coordinates"
                )
                _logger.warning(msg)
                return None, [msg]
            data = await self.cool_seq_tool.ex_g_coords_mapper.genomic_to_tx_segment(
                **kwargs
            )

        if data.errors:
            return None, data.errors

        data.tx_ac = coerce_namespace(data.tx_ac)

        normalized_gene_response = self._normalized_gene(
            data.gene, use_minimal_gene=use_minimal_gene
        )
        if not normalized_gene_response[0] and normalized_gene_response[1]:
            return None, [normalized_gene_response[1]]

        seg_start = data.seg_start
        genomic_start_location = seg_start.genomic_location if seg_start else None
        if genomic_start_location:
            self._add_ids_to_sequence_location(
                genomic_start_location, data.genomic_ac, seq_id_target_namespace
            )

        seg_end = data.seg_end
        genomic_end_location = seg_end.genomic_location if seg_end else None
        if genomic_end_location:
            self._add_ids_to_sequence_location(
                genomic_end_location, data.genomic_ac, seq_id_target_namespace
            )

        return (
            TranscriptSegmentElement(
                transcript=data.tx_ac,
                strand=data.strand,
                # offset by 1 because in CST exons are 0-based
                exonStart=seg_start.exon_ord + 1 if seg_start else None,
                exonStartOffset=seg_start.offset if seg_start else None,
                # offset by 1 because in CST exons are 0-based
                exonEnd=seg_end.exon_ord + 1 if seg_end else None,
                exonEndOffset=seg_end.offset if seg_end else None,
                gene=normalized_gene_response[0],
                elementGenomicStart=genomic_start_location,
                elementGenomicEnd=genomic_end_location,
                coverage=coverage if coverage else None,
                anchoredReads=reads if reads else None,
            ),
            None,
        )

    def gene_element(
        self, gene: str, use_minimal_gene: bool = True
    ) -> tuple[GeneElement | None, str | None]:
        """Create gene element

        :param str gene: The gene symbol name
        :param bool use_minimal_gene: `True` if minimal gene object
            (`id` and `label`) will be used. `False` if
            gene-normalizer's gene object will be used
        :return: GeneElement, warning
        """
        gene_resp = self._normalized_gene(gene, use_minimal_gene=use_minimal_gene)
        if gene_resp[0]:
            return GeneElement(gene=gene_resp[0]), None
        return None, gene_resp[1]

    def templated_sequence_element(
        self,
        start: int,
        end: int,
        sequence_id: str,
        strand: Strand,
        coordinate_type: CoordinateType = CoordinateType.INTER_RESIDUE,
        seq_id_target_namespace: str | None = None,
    ) -> TemplatedSequenceElement:
        """Create templated sequence element

        :param start: Genomic start
        :param end: Genomic end
        :param sequence_id: Chromosome accession for sequence
        :param strand: Strand
        :param coordinate_type: Determines coordinate base used. Must be one of ``residue``
            or ``inter-residue``. Set to ``inter-residue`` by default.
        :param seq_id_target_namespace: If want to use digest for ``sequence_id``, set
            this to the namespace you want the digest for. Otherwise, leave as ``None``.
        :return: Templated Sequence Element
        """
        if coordinate_type == CoordinateType.RESIDUE:
            start -= 1

        region = self._sequence_location(
            start,
            end,
            sequence_id,
            seq_id_target_namespace=seq_id_target_namespace,
        )

        return TemplatedSequenceElement(region=region, strand=strand)

    @staticmethod
    def linker_element(
        sequence: str,
    ) -> tuple[LinkerElement | None, str | None]:
        """Create linker element

        :param sequence: A nucleotide sequence
        :return: Tuple containing a complete Linker element and None if
            successful, or a None value and warning message if unsuccessful
        """
        try:
            upper_seq = sequence.upper()
            seq = sequenceString(upper_seq)
            linker_sequence = LiteralSequenceExpression(
                sequence=seq, id=f"fusor.sequence:{sequence}"
            )
            return LinkerElement(linkerSequence=linker_sequence), None
        except ValidationError as e:
            msg = str(e)
            _logger.warning(msg)
            return None, msg

    @staticmethod
    def multiple_possible_genes_element() -> MultiplePossibleGenesElement:
        """Create a MultiplePossibleGenesElement.

        :return: MultiplePossibleGenesElement
        """
        return MultiplePossibleGenesElement()

    @staticmethod
    def unknown_gene_element() -> UnknownGeneElement:
        """Create UnknownGeneElement

        :return: UnknownGeneElement
        """
        return UnknownGeneElement()

    def functional_domain(
        self,
        status: DomainStatus,
        name: str,
        functional_domain_id: Annotated[str, StringConstraints(pattern=CURIE_REGEX)],
        gene: str,
        sequence_id: str,
        start: int,
        end: int,
        use_minimal_gene: bool = True,
        seq_id_target_namespace: str | None = None,
    ) -> tuple[FunctionalDomain | None, str | None]:
        """Build FunctionalDomain object.

        :param status: Status for domain.  Must be either ``lost`` or ``preserved``
        :param name: Domain name
        :param functional_domain_id: Domain ID
        :param gene: Gene
        :param sequence_id: protein sequence on which provided coordinates are located
        :param start: start position on sequence
        :param end: end position on sequence
        :param use_minimal_gene: ``True`` if minimal gene object (``id``, ``label``) will be used. ``False`` if gene-normalizer's gene
            object will be used
        :param seq_id_target_namespace: If want to use digest for ``sequence_id``, set
            this to the namespace you want the digest for. Otherwise, leave as ``None``.
        :return: Tuple with FunctionalDomain and None value for warnings if
            successful, or a None value and warning message if unsuccessful
        """
        sequence_id_lower = sequence_id.lower()
        if not (sequence_id_lower.startswith("np_")) or (
            sequence_id_lower.startswith("ensp")
        ):
            msg = "Sequence_id must be a protein accession."
            _logger.warning(msg)
            return None, msg

        seq, warning = self.cool_seq_tool.seqrepo_access.get_reference_sequence(
            sequence_id, start, end
        )

        if not seq:
            return None, warning

        gene_descr, warning = self._normalized_gene(
            gene, use_minimal_gene=use_minimal_gene
        )
        if not gene_descr:
            return None, warning

        loc_descr = self._sequence_location(
            start, end, sequence_id, seq_id_target_namespace=seq_id_target_namespace
        )

        try:
            return (
                FunctionalDomain(
                    id=functional_domain_id,
                    label=name,
                    status=status,
                    associatedGene=gene_descr,
                    sequenceLocation=loc_descr,
                ),
                None,
            )
        except ValidationError as e:
            msg = str(e)
            _logger.warning(msg)
            return None, msg

    def regulatory_element(
        self,
        regulatory_class: RegulatoryClass,
        gene: str,
        use_minimal_gene: bool = True,
    ) -> tuple[RegulatoryElement | None, str | None]:
        """Create RegulatoryElement

        :param regulatory_class: one of {"promoter", "enhancer"}
        :param gene: gene term to fetch normalized gene object for
        :param use_minimal_gene: whether to use the minimal gene object
        :return: Tuple with RegulatoryElement instance and None value for warnings if
            successful, or a None value and warning message if unsuccessful
        """
        gene_descr, warning = self._normalized_gene(
            gene, use_minimal_gene=use_minimal_gene
        )
        if not gene_descr:
            return None, warning

        try:
            return (
                RegulatoryElement(
                    regulatoryClass=regulatory_class, associatedGene=gene_descr
                ),
                None,
            )
        except ValidationError as e:
            msg = str(e)
            _logger.warning(msg)
            return None, msg

    def _sequence_location(
        self,
        start: int,
        end: int,
        sequence_id: str,
        seq_id_target_namespace: str | None = None,
    ) -> SequenceLocation:
        """Create SequenceLocation object

        :param start: Start position
        :param end: End position
        :param sequence_id: Accession for sequence
        :param seq_id_target_namespace: If want to use digest for ``sequence_id``, set
            this to the namespace you want the digest for. Otherwise, leave as ``None``.
        """
        sequence_id = self._get_coerced_sequence_id(
            sequence_id, seq_id_target_namespace
        )

        refget_accession = translate_identifier(self.seqrepo, sequence_id)

        sequence_location = SequenceLocation(
            start=start,
            end=end,
            sequenceReference=SequenceReference(
                id=sequence_id, refgetAccession=refget_accession.replace("ga4gh:", "")
            ),
        )
        sequence_location_id = ga4gh_identify(sequence_location)
        sequence_location.id = sequence_location_id

        return sequence_location

    @staticmethod
    def _location_id(
        location: dict,
    ) -> Annotated[str, StringConstraints(pattern=CURIE_REGEX)]:
        """Return GA4GH digest for location

        :param location: VRS Location represented as a dict
        :return: GA4GH digest
        """
        return ga4gh_identify(models.Location(**location))

    def _normalized_gene(
        self, query: str, use_minimal_gene: bool | None = None
    ) -> tuple[MappableConcept | None, str | None]:
        """Return gene from normalized response.

        :param query: A query containing a gene symbol string
        :param use_minimal_gene: bool Use minimal gene representation (id and label only)
        :return: Tuple with gene and None value for warnings if
            successful, and None value with warning string if unsuccessful
        """
        gene_norm_resp = self.gene_normalizer.normalize(query)
        if gene_norm_resp.match_type:
            gene = gene_norm_resp.gene
            if use_minimal_gene:
                return MappableConcept(
                    primaryCoding=gene.primaryCoding,
                    name=gene.name,
                    conceptType=gene.conceptType,
                ), None
            return gene, None
        return None, f"gene-normalizer unable to normalize {query}"

    def _add_ids_to_sequence_location(
        self,
        sequence_location: SequenceLocation,
        genomic_ac: str,
        seq_id_target_namespace: str | None = None,
    ) -> None:
        """Modify the sequence_location to have ga4gh_identified id and its sequenceReference with id from target namespace (refseq default)

        :param sequence_location: the SequenceLocation to add/modify id's of
        :param seq_id_target_namespace: the target namespace for the SequenceLocation's sequenceReference id, which is the genomic_ac
        (defaults to refseq if none given)
        """
        seq_ref_id = self._get_coerced_sequence_id(genomic_ac, seq_id_target_namespace)

        if sequence_location:
            sequence_location.id = ga4gh_identify(sequence_location)
            if sequence_location.sequenceReference:
                sequence_location.sequenceReference.id = seq_ref_id

    def _get_coerced_sequence_id(
        self, sequence_id: str, seq_id_target_namespace: str | None = None
    ) -> str:
        """Get the coerced sequence_id using a target namespace and log any errors

        :param sequence_id: the sequence id to coerce
        :param seq_id_target_namespace: the target namespace
        """
        try:
            sequence_id = coerce_namespace(sequence_id)
        except ValueError:
            if not re.match(CURIE_REGEX, sequence_id):
                sequence_id = f"sequence.id:{sequence_id}"

        if seq_id_target_namespace:
            try:
                seq_id = translate_identifier(
                    self.seqrepo, sequence_id, target_namespace=seq_id_target_namespace
                )
            except IDTranslationException:
                _logger.warning(
                    "Unable to translate %s using %s as the target namespace",
                    sequence_id,
                    seq_id_target_namespace,
                )
            else:
                sequence_id = seq_id
        return sequence_id

    def generate_nomenclature(self, fusion: Fusion) -> str:
        """Generate human-readable nomenclature describing provided fusion

        :param fusion: a valid fusion
        :return: string summarizing fusion in human-readable way per VICC fusion
            curation nomenclature
        """
        return generate_nomenclature(fusion, self.seqrepo)
